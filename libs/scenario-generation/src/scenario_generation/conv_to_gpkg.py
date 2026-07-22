"""
Script for TNTP (original) -> GeoPackage conversion.

This script reads TNTP / GeoJSON network files and converts them into a
single GeoPackage file containing nodes, links, and OD data.
"""

from pathlib import Path

import click
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point


# --- Network reader functions ---
# These functions read the network data from the TNTP (nodes, links, OD) and
# GeoJSON (nodes, fallback) files and return geopandas GeoDataFrames.
# -----------------------------------------------------------------


def read_nodes(network_path: Path, network_name: str) -> gpd.GeoDataFrame:
    """
    Reads node data from either a .tntp or .geojson file, ensuring
    a consistent output format.

    Args:
        network_path: The path to the network directory.
        network_name: The name of the network.

    Returns:
        A GeoDataFrame with columns ['node_id', 'x', 'y', 'geometry'].
    """
    # identifying node file
    node_tntp_path = network_path / "nodes.tntp"
    node_geojson_path = network_path / f"{network_name}_nodes.geojson"

    # reading file
    if node_tntp_path.exists():
        print(f"Reading nodes from {node_tntp_path}")
        nodes_df = pd.read_csv(node_tntp_path, sep="\t", index_col=False)
        # ensuring column consistency
        nodes_df.columns = [col.strip().lower() for col in nodes_df.columns]
        nodes_df = nodes_df.rename(columns={"node": "node_id"})
        nodes_df = nodes_df[["node_id", "x", "y"]]

        # adding geometry
        geometry = [Point(xy) for xy in zip(nodes_df["x"], nodes_df["y"])]
        return gpd.GeoDataFrame(nodes_df, geometry=geometry, crs="EPSG:4326")

    elif node_geojson_path.exists():
        print(f"Reading nodes from {node_geojson_path}")
        nodes_gdf = gpd.read_file(node_geojson_path)

        # find node indentifier column
        for col_name in nodes_gdf.columns:
            if "id" in col_name.lower() or "node" in col_name.lower():
                node_id_col = col_name
                break
        else:
            raise ValueError(
                "No node identifier column found in the geojson file."
                "Ensure there is an attribute with 'id' or 'node' in its name."
            )

        # ensuring column consistency
        nodes_gdf = nodes_gdf.rename(columns={node_id_col: "node_id"})[["node_id", "geometry"]]

        nodes_gdf["x"] = nodes_gdf.geometry.x
        nodes_gdf["y"] = nodes_gdf.geometry.y

        return nodes_gdf[["node_id", "x", "y", "geometry"]]

    else:
        raise FileNotFoundError(
            f"No node file found for network {network_name}. "
            f"Searched for {node_tntp_path} and {node_geojson_path}"
        )


def read_links(network_path: Path, nodes_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Reads link data and creates LineString geometries.

    Args:
        network_path: The path to the network directory.
        nodes_gdf: A GeoDataFrame of the network nodes.

    Returns:
        A GeoDataFrame containing the link data and geometries.
    """
    # reading file
    link_path = network_path / "net.tntp"
    print(f"Reading links from {link_path}")

    skiprows = 0
    with open(link_path, "r") as f:
        for i, line in enumerate(f):
            if line.strip().startswith("~"):
                skiprows = i + 1
                break
    if skiprows == 0:
        raise ValueError("Could not find data section in net.tntp")

    links_df = pd.read_csv(
        link_path,
        skiprows=skiprows,
        sep="\t",
        header=None,
        names=[
            "newline",
            "init_node",
            "term_node",
            "capacity",
            "length",
            "free_flow_time",
            "b",
            "power",
            "speed",
            "toll",
            "link_type",
            "terminator",
        ],
        engine="python",
    )
    # ensuring column consistency
    links_df.drop(columns=["newline", "terminator"], inplace=True)
    links_df.dropna(how="all", inplace=True)

    for col in links_df.columns:
        if links_df[col].dtype == "object":
            links_df[col] = pd.to_numeric(links_df[col], errors="ignore")
    for col in ["init_node", "term_node", "link_type"]:
        links_df[col] = links_df[col].astype(int)

    links_df.insert(0, "link_id", np.arange(1, len(links_df) + 1))

    # adding geometry
    nodes_geom = nodes_gdf.set_index("node_id")["geometry"]
    merged = links_df.merge(
        nodes_geom.rename("orig_geom"), left_on="init_node", right_index=True
    ).merge(nodes_geom.rename("dest_geom"), left_on="term_node", right_index=True)

    geometry = merged.apply(lambda row: LineString([row["orig_geom"], row["dest_geom"]]), axis=1)

    links_gdf = gpd.GeoDataFrame(merged, geometry=geometry, crs="EPSG:4326")
    return links_gdf.drop(columns=["orig_geom", "dest_geom"])


def read_od(network_path: Path, nodes_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Reads origin-destination data and creates LineString geometries.

    Args:
        network_path: The path to the network directory.
        nodes_gdf: A GeoDataFrame of the network nodes.

    Returns:
        A GeoDataFrame containing the OD data and geometries.
    """
    # reading file
    od_path = network_path / "trips.tntp"
    print(f"Reading OD data from {od_path}")

    with open(od_path, "r") as f:
        lines = f.read()

    od_data = []
    for block in lines.split("Origin")[1:]:
        parts = block.strip().split("\n")
        origin_str = parts[0].strip()
        if not origin_str:
            continue
        origin = int(origin_str)

        for line in parts[1:]:
            for pair in line.split(";"):
                if ":" in pair:
                    dest, demand = pair.split(":")
                    if dest.strip() and demand.strip():
                        od_data.append(
                            {
                                "origin": origin,
                                "destination": int(dest.strip()),
                                "demand": float(demand.strip()),
                            }
                        )
    od_df = pd.DataFrame(od_data)

    # adding geometry
    nodes_geom = nodes_gdf.set_index("node_id")["geometry"]
    merged = od_df.merge(nodes_geom.rename("orig_geom"), left_on="origin", right_index=True).merge(
        nodes_geom.rename("dest_geom"), left_on="destination", right_index=True
    )

    geometry = merged.apply(lambda row: LineString([row["orig_geom"], row["dest_geom"]]), axis=1)

    od_gdf = gpd.GeoDataFrame(merged, geometry=geometry, crs="EPSG:4326")
    return od_gdf.drop(columns=["orig_geom", "dest_geom"])


def read_flows(network_path: Path):
    """
    Reads best-known flow data from tntp.
    """
    # reading file
    flows_path = network_path / "flows.tntp"
    print(f"Reading flows from {flows_path}")

    skiprows = 1
    flows_df = pd.read_csv(
        flows_path,
        skiprows=skiprows,
        sep="\t",
        header=None,
        names=["a_node", "b_node", "flow", "cost"],
    )

    flows_df.dropna(how="all", inplace=True)
    flows_gdf = gpd.GeoDataFrame(flows_df, geometry=None)

    return flows_gdf


# --- CLI Definition ---
# This section defines the command-line interface using Click.
# It calls the reader functions and writes the data to a GeoPackage file.
# -----------------------------------------------------------------
@click.command("initialize")
@click.argument("network")
@click.option(
    "--path",
    default="networks",
    show_default=True,
    help="The base path to the networks directory.",
)
def create_gpkg(network: str, path: str = None):
    """
    Creates a GeoPackage for a given network by calling the read_* functions.

    \b
    Args:
        network: The name of the network (e.g., 'anaheim').
        path: The base path of the networks directory.
    """
    # setting up paths
    if path is None:
        path = Path.cwd()
    else:
        path = Path(path)

    network_path = path / network

    if not network_path.is_dir():
        raise ValueError(
            f"Network path {path} does not exist."
            "Make sure to run the script from the network directory or provide a base_path."
        )

    output_path = network_path / f"{network}_master.gpkg"

    print(f"\nProcessing network: {network}")

    # reading data
    nodes_gdf = read_nodes(network_path, network)
    links_gdf = read_links(network_path, nodes_gdf)
    od_gdf = read_od(network_path, nodes_gdf)
    flows_gdf = read_flows(network_path)

    # writing to geopackage
    print(f"Writing to GeoPackage: {output_path}")
    nodes_gdf.to_file(output_path, layer="nodes", driver="GPKG")
    links_gdf.to_file(output_path, layer="links", driver="GPKG")
    od_gdf.to_file(output_path, layer="od", driver="GPKG")
    flows_gdf.to_file(output_path, layer="flows", driver="GPKG")

    print("\nDone.")
