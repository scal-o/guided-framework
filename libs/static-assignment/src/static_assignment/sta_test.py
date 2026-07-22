"""
Script for Static Traffic Assignment Verification & Convergence Testing

This module validates the accuracy and convergence of AequilibraE's User Equilibrium (UE)
assignment results (for scenario_00000) against ground-truth flow benchmarks from TNTP files.

Key Evaluation Metrics:
1. wMAPE (Weighted Mean Absolute Percentage Error across all links).
2. % of Links with Relative Error <= 5% and <= 10% (filtered for links with flow > 10).
3. Flow-Weighted Pass Rate (% of total baseline flow mass carried by links within 5% error).
"""

from pathlib import Path
from typing import Optional

import click
import geopandas as gpd


@click.command("test")
@click.argument("network")
@click.option(
    "--path",
    default="data",
    show_default=True,
    help="The base path to the scenarios directory.",
)
def test_sta(
    network: str,
    path: Optional[str] = None,
):
    """
    Validates AequilibraE static assignment results for NETWORK (scenario_00000)
    against ground-truth flow benchmarks from TNTP repository files.
    """
    print("--- Static Traffic Assignment Benchmark Verification ---")
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

    if not output_path.is_dir():
        raise FileNotFoundError(
            f"Output path {output_path} does not exist. Run 'sta run' before testing."
        )

    base_scenario = scenarios_path / "scenarios_geojson" / "scenario_00000"
    print(f"\nEvaluating scenario_00000 baseline assignment against TNTP benchmark flows...")

    # Load ground-truth flows (TNTP benchmark) and own assignment predictions (Parquet)
    best_known_flows = gpd.read_file(base_scenario / "flows.geojson")
    sta_own_flows = gpd.read_parquet(output_path / "scenario_00000" / "scenario_00000.parquet")

    # Verify link counts match
    if len(best_known_flows) != len(sta_own_flows):
        raise ValueError(
            f"Link count mismatch: benchmark has {len(best_known_flows)} links, "
            f"own assignment has {len(sta_own_flows)} links."
        )

    # Sort dataframes by (a_node, b_node) pair for exact alignment
    best_known_flows.sort_values(
        by=["a_node", "b_node"], ascending=True, inplace=True, ignore_index=True
    )
    sta_own_flows.sort_values(
        by=["a_node", "b_node"], ascending=True, inplace=True, ignore_index=True
    )

    best_known_flows = best_known_flows.astype({"a_node": "int32", "b_node": "int32"})
    sta_own_flows = sta_own_flows.astype({"a_node": "int32", "b_node": "int32"})

    # Verify identical topology
    if not best_known_flows[["a_node", "b_node"]].equals(sta_own_flows[["a_node", "b_node"]]):
        raise ValueError("Link alignment mismatch: link (a_node, b_node) pairs do not match.")

    # Extract flow vectors
    best_known_flows = best_known_flows["flow"].astype("float64")
    sta_own_flows = sta_own_flows["flow"].astype("float64")

    # =========================================================================
    # Metrics Calculation
    # =========================================================================

    # Absolute flow differences per link
    diff = (sta_own_flows - best_known_flows).abs()

    # 1. wMAPE (Weighted Mean Absolute Percentage Error across all links)
    total_baseline_flow = float(best_known_flows.abs().sum())
    if total_baseline_flow <= 0:
        raise ValueError("Baseline flows sum to 0; cannot compute wMAPE.")
    wmape = float(diff.sum() / total_baseline_flow)

    # 2. Filtered link-level relative error pass rates (links with baseline flow > 10)
    cutoff = 10.0
    mask = best_known_flows > cutoff
    n_total = int(len(best_known_flows))
    n_filtered = int(mask.sum())

    if n_filtered == 0:
        raise ValueError(
            f"No links with baseline flow > {cutoff}. Cannot compute link-level pass rates."
        )

    rel_err = (diff[mask] / best_known_flows[mask]).astype("float64")
    pct_within_5 = float((rel_err <= 0.05).mean())
    pct_within_10 = float((rel_err <= 0.10).mean())

    # 3. Flow-weighted share within 5%: fraction of total flow mass on links within 5% error
    baseline_filtered = best_known_flows[mask].astype("float64")
    within_5_mask = rel_err <= 0.05
    flow_weighted_within_5 = float(baseline_filtered[within_5_mask].sum() / baseline_filtered.sum())

    # =========================================================================
    # Formatted Summary Output
    # =========================================================================
    print("\n" + "=" * 55)
    print("        STATIC TRAFFIC ASSIGNMENT VERIFICATION        ")
    print("=" * 55)
    print(f"Links Evaluated:           {n_filtered} / {n_total} (flow > {cutoff:.0f})")
    print(f"Links Within 5% Rel. Error: {pct_within_5 * 100:.2f}%")
    print(f"Links Within 10% Rel. Error:{pct_within_10 * 100:.2f}%")
    print(f"Flow-Weighted Within 5%:   {flow_weighted_within_5 * 100:.2f}%")
    print(f"Overall wMAPE (All Links):  {wmape * 100:.2f}%")
    print("-" * 55)

    # Pass / Fail criteria for mathematical convergence:
    # - >= 95% flow mass on links within 5% relative error
    # - <= 5% overall weighted error (wMAPE)
    flow_weighted_threshold = 0.95
    wmape_threshold = 0.05

    if flow_weighted_within_5 >= flow_weighted_threshold and wmape <= wmape_threshold:
        print(
            "RESULT: PASS [Convergence thresholds satisfied: "
            f"Flow-weighted >= {flow_weighted_threshold * 100:.0f}%, "
            f"wMAPE <= {wmape_threshold * 100:.0f}%]"
        )
    else:
        print(
            "RESULT: FAIL [Convergence thresholds not satisfied: "
            f"Flow-weighted >= {flow_weighted_threshold * 100:.0f}%, "
            f"wMAPE <= {wmape_threshold * 100:.0f}%]"
        )
    print("=" * 55 + "\n")
