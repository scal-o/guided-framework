"""
Script for GeoJSON <-> OpenMatrix (AequilibraE-compatible format) conversion utilities.

This module provides:

- Conversion of per-scenario OD tables from `od.geojson` to `od.omx`
- Cleanup of old `od.geojson` files once `od.omx` is present and valid
- Conversion back from `od.omx` to `od.geojson`
"""

from multiprocessing import Pool, cpu_count
from pathlib import Path

import click
import geopandas as gpd
import numpy as np
import openmatrix as omx
import pandas as pd
from tqdm import tqdm


def _scenarios_path(network: str, path: str | None) -> Path:
    """Builds the scenarios directory path for a given network.

    Args:
        network: Network name under the base path.
        path: Base data path. If None, uses current working directory.

    Returns:
        Path to `<base>/<network>/scenarios_geojson`.
    """
    base_path = Path.cwd() if path is None else Path(path)
    return base_path / network / "scenarios_geojson"


def _get_scenario_dirs(network: str, path: str | None) -> list[Path]:
    """Discovers scenario directories under the scenarios path.

    Args:
        network: Network name under the base path.
        path: Base data path. If None, uses current working directory.

    Returns:
        A list of `scenario_*` directories.

    Raises:
        ValueError: If scenarios directory does not exist.
        FileNotFoundError: If no scenario directories are found.
    """
    scenarios_path = _scenarios_path(network=network, path=path)

    if not scenarios_path.is_dir():
        raise ValueError(
            f"Scenarios path {scenarios_path} does not exist. "
            "Make sure to run the script from the data directory or provide a base path."
        )

    scenario_dirs = list(scenarios_path.glob("scenario_*"))
    if not scenario_dirs:
        raise FileNotFoundError(f"No scenario files found in {scenarios_path}")

    return scenario_dirs


def _is_valid_omx(scenario_dir: Path) -> bool:
    """Checks whether `od.omx` exists and looks like a complete conversion output.

    Validation criteria:
    - `od.omx` file exists
    - contains a matrix named `"matrix"`
    - contains a mapping `"taz"`
    - matrix shape matches mapping size (N x N)

    Args:
        scenario_dir: Scenario directory containing `od.omx`.

    Returns:
        True if the file appears valid, otherwise False.
    """
    output_file = scenario_dir / "od.omx"
    if not output_file.is_file():
        return False

    try:
        with omx.open_file(str(output_file), "r") as omx_file:
            if "matrix" not in omx_file.list_matrices():
                return False
            if "taz" not in omx_file.list_mappings():
                return False

            matrix = omx_file["matrix"][:]
            taz_mapping = omx_file.mapping("taz")
            n = len(taz_mapping)

            if matrix.ndim != 2:
                return False
            if matrix.shape != (n, n):
                return False

        return True
    except Exception:
        return False


def convert_od(scenario_dir: Path) -> bool:
    """Converts one scenario's OD table from `od.geojson` into `od.omx`.

    Args:
        scenario_dir: Scenario directory containing `od.geojson`.

    Returns:
        True on success, False on failure.
    """
    output_file = scenario_dir / "od.omx"

    try:
        od_df = gpd.read_file(scenario_dir / "od.geojson")
        od_df = od_df[["origin", "destination", "demand"]]

        zone_ids = pd.concat([od_df["origin"], od_df["destination"]])
        zone_ids = sorted(zone_ids.unique())

        od_pivot = od_df.pivot_table(
            index="origin", columns="destination", values="demand", fill_value=0.0
        )

        matrix_full = od_pivot.reindex(index=zone_ids, columns=zone_ids, fill_value=0.0)
        matrix_data = matrix_full.to_numpy(dtype=np.float32)

        with omx.open_file(str(output_file), "w") as omx_file:
            omx_file["matrix"] = matrix_data
            omx_file.create_mapping("taz", zone_ids)

        return True

    except Exception as e:
        print(f"Error processing {scenario_dir.name}: {e}")
        return False


def cleanup_geojson_od(scenario_dir: Path) -> bool:
    """Deletes `od.geojson` if `od.omx` exists and appears valid.

    Args:
        scenario_dir: Scenario directory.

    Returns:
        True if `od.geojson` was deleted, False otherwise (skipped or failed).
    """
    geojson_path = scenario_dir / "od.geojson"

    if not geojson_path.is_file():
        return False

    if not _is_valid_omx(scenario_dir):
        return False

    try:
        geojson_path.unlink()
        return True
    except Exception as e:
        print(f"Error deleting {geojson_path}: {e}")
        return False


def convert_od_back_to_geojson(scenario_dir: Path) -> bool:
    """Converts one scenario's OD matrix from `od.omx` back into `od.geojson`.

    This will overwrite an existing `od.geojson` (if present).

    Args:
        scenario_dir: Scenario directory.

    Returns:
        True on success, False on failure.
    """
    omx_path = scenario_dir / "od.omx"
    geojson_path = scenario_dir / "od.geojson"

    if not omx_path.is_file():
        return False

    try:
        with omx.open_file(str(omx_path), "r") as omx_file:
            if "matrix" not in omx_file.list_matrices():
                return False
            if "taz" not in omx_file.list_mappings():
                return False

            matrix = omx_file["matrix"][:]
            taz_mapping = omx_file.mapping("taz")

        # OpenMatrix mapping maps zone_id -> index. We want an index-ordered list of zone ids.
        zone_ids = [zone_id for zone_id, _idx in sorted(taz_mapping.items(), key=lambda kv: kv[1])]

        # Convert to long format and drop zeros (keep only positive demand).
        df = pd.DataFrame(matrix, index=zone_ids, columns=zone_ids)
        long_df = df.stack().reset_index()
        long_df.columns = ["origin", "destination", "demand"]
        long_df = long_df[long_df["demand"] > 0.0]

        # GeoJSON without geometry: store as a GeoDataFrame with empty geometry.
        gdf = gpd.GeoDataFrame(long_df, geometry=None, crs=None)
        gdf.to_file(geojson_path, driver="GeoJSON")

        return True
    except Exception as e:
        print(f"Error processing {scenario_dir.name}: {e}")
        return False


@click.group("convert")
def convert_cli() -> None:
    """Conversion utilities for scenario OD matrices."""
    pass


@convert_cli.command("to-omx")
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
    help="Number of parallel processes to use. Defaults to number of CPU cores.",
)
def convert_to_omx(
    network: str,
    path: str = None,
    processes: int = None,
) -> None:
    """Converts each `od.geojson` for NETWORK into its own `od.omx` file."""
    print("--- OpenMatrix conversion ---")
    print(f"Network: {network}")
    print(f"Path: {path}")

    scenario_dirs = _get_scenario_dirs(network=network, path=path)

    print(f"Found {len(scenario_dirs)} scenario files to process.")
    print(f"Starting conversion of {len(scenario_dirs)} files to .omx format...")

    if processes is None:
        processes = cpu_count() - 2

    with Pool(processes=processes) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(convert_od, scenario_dirs),
                total=len(scenario_dirs),
                desc="Processing & Writing .omx",
            )
        )

    print("--- OpenMatrix File Creation Complete ---")
    success_count = sum(1 for res in results if res)
    print(f"Successfully wrote {success_count} / {len(scenario_dirs)} files.")


@convert_cli.command("cleanup")
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
    help="Number of parallel processes to use. Defaults to number of CPU cores.",
)
def cleanup_geojson(
    network: str,
    path: str = None,
    processes: int = None,
) -> None:
    """Deletes `od.geojson` per scenario if `od.omx` exists and appears valid."""
    print("--- Cleanup old GeoJSON OD tables ---")
    print(f"Network: {network}")
    print(f"Path: {path}")

    scenario_dirs = _get_scenario_dirs(network=network, path=path)

    if processes is None:
        processes = cpu_count() - 2

    with Pool(processes=processes) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(cleanup_geojson_od, scenario_dirs),
                total=len(scenario_dirs),
                desc="Deleting od.geojson (when possible)",
            )
        )

    deleted_count = sum(1 for res in results if res)
    print("--- Cleanup Complete ---")
    print(f"Deleted {deleted_count} / {len(scenario_dirs)} od.geojson files.")


@convert_cli.command("to-geojson")
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
    help="Number of parallel processes to use. Defaults to number of CPU cores.",
)
def convert_to_geojson(
    network: str,
    path: str = None,
    processes: int = None,
) -> None:
    """Converts each `od.omx` for NETWORK back into `od.geojson`."""
    print("--- Convert OMX back to GeoJSON OD tables ---")
    print(f"Network: {network}")
    print(f"Path: {path}")

    scenario_dirs = _get_scenario_dirs(network=network, path=path)

    if processes is None:
        processes = cpu_count() - 2

    with Pool(processes=processes) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(convert_od_back_to_geojson, scenario_dirs),
                total=len(scenario_dirs),
                desc="Writing od.geojson",
            )
        )

    success_count = sum(1 for res in results if res)
    print("--- Conversion Complete ---")
    print(f"Successfully wrote {success_count} / {len(scenario_dirs)} od.geojson files.")
