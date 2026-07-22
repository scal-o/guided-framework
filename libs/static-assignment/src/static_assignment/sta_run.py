"""
Script for Static Traffic Assignment (STA) Execution

This module executes the macroscopic User Equilibrium (UE) Traffic Assignment Problem (TAP)
on scenario networks using AequilibraE's Bi-Conjugate Frank-Wolfe (BFW) algorithm.
The resulting equilibrium flows, Volume-to-Capacity Ratios (V/C), and travel times
are saved as Parquet datasets for training and evaluating GNN surrogate models.
"""

import io
import logging
import shutil
from contextlib import redirect_stderr
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Optional

import click
import geopandas as gpd
import openmatrix as omx
import pandas as pd
from aequilibrae.matrix import AequilibraeMatrix
from aequilibrae.paths import Graph
from aequilibrae.paths.traffic_assignment import TrafficAssignment
from aequilibrae.paths.traffic_class import TrafficClass
from tqdm import tqdm

from static_assignment.od_convert import convert_od

# Suppress verbose log warnings from AequilibraE internal routines
logger = logging.getLogger("aequilibrae")
logger.setLevel(logging.ERROR)

# Mapping for network-specific centroid flow blocking behavior.
# Anaheim requires blocking centroid flows through TAZ nodes; Chicago Sketch does not.
CENTROID_FLOW_BLOCKING = {"anaheim": True, "chicago": False}


def find_centroids(scenario_path: Path) -> pd.Series:
    """
    Extracts zone centroid IDs ordered by index from scenario's OMX demand file.
    If `od.omx` does not exist yet, automatically converts `od.geojson` to `od.omx`.

    Args:
        scenario_path: Path to scenario directory (e.g. data/anaheim/scenarios_geojson/scenario_00000).

    Returns:
        pd.Series containing the ordered centroid TAZ IDs.
    """
    omx_file_path = scenario_path / "od.omx"
    geojson_file_path = scenario_path / "od.geojson"

    # Transparently convert od.geojson -> od.omx if openmatrix file is missing
    if not omx_file_path.exists():
        if geojson_file_path.exists():
            convert_od(scenario_path)
        else:
            raise FileNotFoundError(f"Demand data for scenario '{scenario_path.name}' not found.")

    # Read the TAZ mapping from the OMX file
    with omx.open_file(str(omx_file_path), "r") as omx_file:
        if "taz" not in omx_file.list_mappings():
            raise FileNotFoundError(
                f"TAZ mapping not found in OMX file for scenario '{scenario_path.name}'."
            )

        taz_mapping = omx_file.mapping("taz")
        # Extract zone IDs sorted by their index in the OpenMatrix mapping
        centroids_list = [
            zone_id for zone_id, _idx in sorted(taz_mapping.items(), key=lambda kv: kv[1])
        ]

    return pd.Series(centroids_list)


def load_network(scenario_path: Path, block_centroid_flows: bool = False) -> Graph:
    """
    Loads link and node geometry GeoJSON files for a scenario and prepares an AequilibraE Graph.

    Args:
        scenario_path: Path to scenario directory.
        block_centroid_flows: Whether to prevent traffic from routing directly through centroids.

    Returns:
        Instantiated and prepared AequilibraE Graph instance.
    """
    # Verify input GeoJSON files exist
    if not scenario_path.exists():
        raise FileNotFoundError(f"Scenario '{scenario_path.name}' data not found.")
    elif not (scenario_path / "links.geojson").exists():
        raise FileNotFoundError(f"Network links for scenario '{scenario_path.name}' not found.")
    elif not (scenario_path / "nodes.geojson").exists():
        raise FileNotFoundError(f"Nodes for scenario '{scenario_path.name}' not found.")

    # Retrieve network centroids for path finding
    centroids = find_centroids(scenario_path)

    # Load link attributes and rename node columns to AequilibraE convention (a_node, b_node)
    net_df = gpd.read_file(scenario_path / "links.geojson")
    net_df = net_df[
        [
            "link_id",
            "init_node",
            "term_node",
            "capacity",
            "free_flow_time",
            "b",
            "power",
            "geometry",
        ]
    ]
    net_df = net_df.rename(columns={"init_node": "a_node", "term_node": "b_node"})
    net_df = net_df.assign(direction=1)

    # Load node coordinates
    nodes = gpd.read_file(scenario_path / "nodes.geojson")
    nodes = nodes[["node_id", "x", "y"]]
    nodes.index = nodes["node_id"].copy()
    nodes = nodes.rename(columns={"x": "lat", "y": "lon"})

    # Instantiate AequilibraE graph and bind capacity/free-flow time attributes
    graph = Graph()
    graph.network = net_df
    graph.prepare_graph(centroids)

    graph.set_graph("free_flow_time")
    graph.capacity = net_df["capacity"].values
    graph.free_flow_time = net_df["free_flow_time"].values
    graph.set_blocked_centroid_flows(block_centroid_flows)
    graph.network["id"] = graph.network["link_id"]
    graph.lonlat_index = nodes.loc[graph.all_nodes]

    return graph


def load_matrix(scenario_path: Path) -> AequilibraeMatrix:
    """
    Loads an AequilibraE matrix object from an OMX demand file.
    Automatically converts od.geojson to od.omx if needed.

    Args:
        scenario_path: Path to scenario directory.

    Returns:
        Prepared AequilibraeMatrix instance.
    """
    omx_path = scenario_path / "od.omx"
    geojson_path = scenario_path / "od.geojson"

    # Transparently convert od.geojson -> od.omx if openmatrix file is missing
    if not omx_path.exists():
        if geojson_path.exists():
            convert_od(scenario_path)
        else:
            raise FileNotFoundError(f"Demand matrix for scenario '{scenario_path.name}' not found.")

    mat = AequilibraeMatrix()
    mat.create_from_omx(str(omx_path))
    mat.computational_view(["matrix"])
    return mat


def run_assignment(scenario_name: str, scenarios_dir: Path, block_centroid_flows: bool = False) -> bool:
    """
    Executes a single static traffic assignment run for one scenario directory.

    1. Loads network graph and demand matrix.
    2. Instantiates AequilibraE TrafficAssignment with BPR volume-delay functions.
    3. Runs Bi-Conjugate Frank-Wolfe (BFW) solver until rgap <= 1e-5 or max_iter=500.
    4. Serializes output flows, V/C ratios, and convergence logs to Parquet format.

    Args:
        scenario_name: Directory name of scenario (e.g. 'scenario_00001').
        scenarios_dir: Base directory containing scenario folders.
        block_centroid_flows: Whether to block centroid flow routing.

    Returns:
        True on successful completion.
    """
    scenario_path = scenarios_dir / "scenarios_geojson" / f"{scenario_name}"
    output_path = scenarios_dir / "scenarios_sta_results" / f"{scenario_name}"
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Load network graph and OD matrix
    graph = load_network(scenario_path, block_centroid_flows)
    matrix = load_matrix(scenario_path)

    # 2. Create traffic class container
    traffic_class = TrafficClass("c", graph, matrix)

    # 3. Configure AequilibraE BFW assignment solver
    assignment = TrafficAssignment()
    assignment.set_classes([traffic_class])
    assignment.set_vdf("BPR")
    assignment.set_vdf_parameters({"alpha": "b", "beta": "power"})
    assignment.set_capacity_field("capacity")
    assignment.set_time_field("free_flow_time")
    assignment.set_algorithm("bfw")  # Bi-Conjugate Frank-Wolfe
    assignment.max_iter = 500        # Iteration ceiling
    assignment.rgap_target = 1e-5    # Target relative gap for equilibrium convergence
    assignment.set_cores(1)          # Run single-threaded inside multiprocessing worker

    # 4. Execute solver redirecting stderr logging
    with io.StringIO() as s:
        with redirect_stderr(s):
            assignment.execute()

    # 5. Extract assignment metrics (link flow, V/C, congested travel time)
    results = assignment.results()
    network = graph.network.copy()
    network.index = network["id"].values

    network["flow"] = results["PCE_AB"]
    network["volume_capacity_ratio"] = results["VOC_AB"]
    network["congested_time"] = results["Congested_Time_AB"]

    # 6. Save results and convergence report to Parquet files
    scenario_results_path = output_path / f"{scenario_name}.parquet"
    scenario_convergence_path = output_path / f"{scenario_name}_convergence.parquet"

    network.to_parquet(scenario_results_path)
    assignment.report().to_parquet(scenario_convergence_path)

    return True


@click.command("run")
@click.argument("network")
@click.option(
    "--path",
    default="data",
    show_default=True,
    help="The base path to the scenarios directory.",
)
@click.option(
    "-p",
    "--processes",
    type=int,
    default=None,
    help="Number of parallel processes to use. Defaults to CPU count - 2.",
)
@click.option(
    "--overwrite",
    is_flag=True,
    default=False,
    show_default=True,
    help="Overwrite existing output directory if present.",
)
def run_sta(
    network: str,
    path: Optional[str] = None,
    processes: Optional[int] = None,
    overwrite: bool = False,
):
    """
    Runs static traffic assignment using AequilibraE (BFW, rgap 1e-5) for all scenarios of NETWORK.
    """
    print("--- Static Traffic Assignment (AequilibraE BFW) ---")
    print(f"Network: {network}")
    print(f"Path: {path}")

    # Set up data paths
    if path is None:
        base_path = Path.cwd()
    else:
        base_path = Path(path)

    scenarios_path = base_path / network
    output_path = scenarios_path / "scenarios_sta_results"

    # Validate directory structure
    if not scenarios_path.is_dir():
        raise ValueError(f"Scenarios path {scenarios_path} does not exist.")

    if not (scenarios_path / "scenarios_geojson").is_dir():
        raise ValueError(f"Scenarios path {scenarios_path / 'scenarios_geojson'} does not exist.")

    # Handle overwrite flag or raise FileExistsError if output path exists
    if output_path.is_dir():
        if overwrite:
            print(f"Removing existing output directory: {output_path}")
            shutil.rmtree(output_path)
        else:
            raise FileExistsError(
                f"Output directory {output_path} already exists. Use --overwrite to replace it."
            )

    output_path.mkdir(parents=True, exist_ok=True)

    # Discover scenario directories
    scenario_dirs = list((scenarios_path / "scenarios_geojson").glob("scenario_*"))
    scenario_names = [scenario.name for scenario in scenario_dirs]

    print(f"Found {len(scenario_names)} scenario directories to process.")

    # Determine centroid flow blocking strategy (matches network prefixes like 'anaheim_a', 'anaheim_b')
    block_centroids = False
    for key, should_block in CENTROID_FLOW_BLOCKING.items():
        if key in network.lower():
            block_centroids = should_block
            break

    worker_func = partial(
        run_assignment,
        scenarios_dir=scenarios_path,
        block_centroid_flows=block_centroids,
    )

    # Determine process pool size
    if processes is None:
        num_processes = cpu_count() - 2
    else:
        num_processes = processes

    num_processes = max(1, min(num_processes, len(scenario_names)))
    print(f"Using {num_processes} parallel processes...")

    # Execute batch assignments in parallel
    with Pool(processes=num_processes) as pool:
        results = list(
            tqdm(
                pool.imap(worker_func, scenario_names),
                total=len(scenario_names),
                desc="Running STA (BFW) & Saving Results",
            )
        )

    success_count = sum(1 for res in results if res)
    print("--- Assignment Complete ---")
    print(f"Successfully processed {success_count} / {len(scenario_names)} static assignments.")
    print(f"Results saved to: {output_path}")
