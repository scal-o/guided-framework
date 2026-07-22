"""
Script for Scenario Generation

Reads a master GeoPackage file for a network and generates N scenario GeoJSON files
for Dataset A (Uniform Scaling) or Dataset B (Dirichlet Distribution Shift).
"""

from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Optional

import click
import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm import tqdm

# --- Perturbation Functions ---


def modify_capacity_uniform(x: pd.Series) -> pd.Series:
    """
    Applies a stochastic capacity reduction factor delta_r ~ U(0.5, 1.0) to link capacity.
    """
    factors = np.random.uniform(0.5, 1.0, size=len(x))
    return (x * factors).round(0).astype(int)


def modify_od_uniform(x: pd.Series) -> pd.Series:
    """
    Applies a stochastic modification factor delta_ij ~ U(0.5, 1.5) to OD demand (Dataset A).
    """
    factors = np.random.uniform(0.5, 1.5, size=len(x))
    return np.ceil(x * factors).astype(int)


def create_od_dirichlet(x: pd.Series, alpha: float = 0.05) -> pd.Series:
    """
    Generates a synthetic origin-destination (OD) matrix using a Dirichlet distribution (Dataset B).
    Redistributes total network demand D_tot across active OD pairs using Dir(alpha * 1_K).
    """
    active_pairs = x > 0
    x_active = x[active_pairs]

    total_demand = float(x_active.sum())
    proportions = np.random.dirichlet(alpha * np.ones(len(x_active)))
    new_x = proportions * total_demand
    res = x.copy()
    res.iloc[active_pairs] = new_x
    return pd.Series(np.ceil(res).astype(int))


# --- Scenario Generator Class Definition ---


class ScenarioGenerator:
    """
    Loads a master network and generates stochastic scenarios.
    """

    def __init__(self, network: str, path: Path):
        self.network_name: str = network
        self.network_path: Path = path / network
        self.master_gpkg_file: Path = self.network_path / f"{self.network_name}_master.gpkg"

        print(f"Loading master network from: {self.master_gpkg_file}")
        self._load_master_data()
        print(
            f"Loaded:\n"
            f"- {len(self.base_nodes_gdf)} master nodes\n"
            f"- {len(self.base_links_gdf)} master links\n"
            f"- {len(self.base_od_gdf)} master OD pairs\n"
            f"- {len(self.base_flows_gdf)} master flows\n"
        )

    def _load_master_data(self) -> None:
        if not self.master_gpkg_file.exists():
            raise FileNotFoundError(f"Master file not found: {self.master_gpkg_file}")
        try:
            self.base_nodes_gdf: gpd.GeoDataFrame = gpd.read_file(
                self.master_gpkg_file, layer="nodes"
            )
            self.base_links_gdf: gpd.GeoDataFrame = gpd.read_file(
                self.master_gpkg_file, layer="links"
            )
            self.base_od_gdf: gpd.GeoDataFrame = gpd.read_file(self.master_gpkg_file, layer="od")
            self.base_flows_gdf: gpd.GeoDataFrame = gpd.read_file(
                self.master_gpkg_file, layer="flows"
            )

            self.base_flows_gdf = gpd.GeoDataFrame(self.base_flows_gdf, geometry=None)

        except Exception as e:
            raise Exception(
                "Failed to read layers from master GeoPackage. "
                "Ensure the file contains 'nodes', 'links', 'od', and 'flows' layers."
            ) from e

    @staticmethod
    def _generate_single_scenario(
        scenario_seed: int,
        base_nodes_gdf: gpd.GeoDataFrame,
        base_links_gdf: gpd.GeoDataFrame,
        base_od_gdf: gpd.GeoDataFrame,
        base_flows_gdf: gpd.GeoDataFrame,
        output_dir: Path,
        dataset_type: str = "dataset_a",
    ) -> bool:
        # Set seed for exact reproducibility per scenario
        np.random.seed(scenario_seed)

        mod_links_gdf = base_links_gdf.copy()
        mod_od_gdf = base_od_gdf.copy()

        # Scenario 0 is baseline (unmodified)
        if scenario_seed != 0:
            # Capacity perturbation U(0.5, 1.0)
            if "capacity" in mod_links_gdf.columns:
                mod_links_gdf["capacity"] = modify_capacity_uniform(mod_links_gdf["capacity"])

            # Demand perturbation
            if dataset_type == "dataset_b":
                mod_od_gdf["demand"] = create_od_dirichlet(mod_od_gdf["demand"], alpha=0.05)
            else:
                mod_od_gdf["demand"] = modify_od_uniform(mod_od_gdf["demand"])

        scenario_filename = f"scenario_{scenario_seed:05d}"
        scenario_output_dir = output_dir / scenario_filename
        scenario_output_dir.mkdir(parents=True, exist_ok=False)

        try:
            base_nodes_gdf.to_file(scenario_output_dir / "nodes.geojson", driver="GeoJSON")
            mod_links_gdf.to_file(scenario_output_dir / "links.geojson", driver="GeoJSON")
            mod_od_gdf.to_file(scenario_output_dir / "od.geojson", driver="GeoJSON")

            if scenario_seed == 0:
                base_flows_gdf.to_file(scenario_output_dir / "flows.geojson", driver="GeoJSON")

        except Exception as e:
            raise Exception(f"Error while saving {scenario_filename}: {e}")

        return True

    def run(
        self,
        n_scenarios: int,
        output_dir: Path,
        dataset_type: str = "dataset_a",
        multiprocess: bool = False,
    ) -> None:
        print(f"\nStarting generation of {n_scenarios} scenarios (Dataset: {dataset_type})...")
        output_dir.mkdir(parents=True, exist_ok=False)

        worker_func = partial(
            ScenarioGenerator._generate_single_scenario,
            base_nodes_gdf=self.base_nodes_gdf,
            base_links_gdf=self.base_links_gdf,
            base_od_gdf=self.base_od_gdf,
            base_flows_gdf=self.base_flows_gdf,
            output_dir=output_dir,
            dataset_type=dataset_type.lower(),
        )

        scenario_ids = list(range(0, n_scenarios + 1))

        if multiprocess:
            num_processes = cpu_count() - 2
            num_processes = min(num_processes, len(scenario_ids))
            print(f"Using {num_processes} parallel processes...")

            with Pool(processes=num_processes) as pool:
                results = list(
                    tqdm(
                        pool.imap(worker_func, scenario_ids),
                        total=len(scenario_ids),
                        desc="Generating Scenarios",
                    )
                )
        else:
            print("Running sequentially (single process)...")
            results = []
            for scenario_id in tqdm(
                scenario_ids, total=len(scenario_ids), desc="Generating Scenarios"
            ):
                result = worker_func(scenario_id)
                results.append(result)

        print("--- Scenario Generation Complete ---")
        print(f"Successfully wrote {len(results)} scenarios to {output_dir}")


# --- CLI Definition ---


@click.command("generate")
@click.argument("network")
@click.option(
    "--path",
    default="networks",
    show_default=True,
    help="The base path to the networks directory.",
)
@click.option(
    "--output",
    default=None,
    show_default=True,
    help="Directory to save generated scenario files. Defaults to data/{network}_{dataset_type_suffix}/.",
)
@click.option(
    "-n",
    "--n-scenarios",
    type=int,
    default=5000,
    show_default=True,
    help="Number of scenarios to generate.",
)
@click.option(
    "--dataset-type",
    type=click.Choice(["dataset_a", "dataset_b"], case_sensitive=False),
    default="dataset_a",
    show_default=True,
    help="Dataset perturbation type: dataset_a (uniform) or dataset_b (dirichlet).",
)
@click.option(
    "--multiprocess",
    is_flag=True,
    default=False,
    show_default=True,
    help="Use multiprocessing for parallel scenario generation.",
)
def generate_scenarios(
    network: str,
    path: Optional[str] = None,
    output: Optional[str] = None,
    n_scenarios: int = 5000,
    dataset_type: str = "dataset_a",
    multiprocess: bool = False,
):
    """
    Generates N stochastic scenarios from a NETWORK's master GeoPackage file.
    """
    print("--- Scenario Generation ---")
    print(f"Network: {network}")
    print(f"Dataset type: {dataset_type}")
    print(f"Base path: {path}")
    print(f"Output directory: {output}")
    print(f"Multiprocessing: {multiprocess}")

    if path is None:
        path_path = Path.cwd()
    else:
        path_path = Path(path)

    network_path = path_path / network

    ds_suffix = "a" if dataset_type.lower() == "dataset_a" else "b"

    if output is None:
        output_path = Path.cwd() / "data" / f"{network}_{ds_suffix}" / "scenarios_geojson"
    else:
        output_path = Path(output) / "scenarios_geojson"

    if not network_path.is_dir():
        raise ValueError(
            f"Network path {path} does not exist."
            "Make sure to run the script from the network directory or provide a base_path."
        )

    if output_path.exists():
        raise FileExistsError(
            f"Output directory {output_path} already exists."
            "Please remove it or choose a different output path."
        )

    try:
        generator = ScenarioGenerator(network, path_path)
        generator.run(
            n_scenarios=n_scenarios,
            output_dir=output_path,
            dataset_type=dataset_type,
            multiprocess=multiprocess,
        )

    except Exception as e:
        raise Exception(f"Scenario generation failed. An unexpected error occurred: {e}") from e
