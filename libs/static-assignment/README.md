# `static-assignment`

A Python package for running macroscopic User Equilibrium (UE) Static Traffic Assignment (STA) on scenario networks using the [AequilibraE](https://www.aequilibrae.com/) framework.

This package computes ground-truth equilibrium link flows, Volume-to-Capacity Ratios (V/C), and congested travel times used for training and evaluating GNN surrogate models as described in **"GUIDED Network-Agnostic Feature Initialization for Spatial Transferability in GNN-based Models"** (Scalese et al., 2026).

---

## Key Features

1. **Static Traffic Assignment (`sta run`)**:
   - Solves the deterministic Traffic Assignment Problem (TAP) using the Bi-Conjugate Frank-Wolfe (BFW) algorithm.
   - Enforces strict equilibrium convergence criteria ($\text{rgap} \le 10^{-5}$, $\text{max\_iter} = 500$).
   - Handles centroid flow blocking dynamically based on network topology (e.g. Anaheim: `True`, Chicago Sketch: `False`).
   - Exports per-scenario link results (`.parquet`) and algorithm convergence reports (`_convergence.parquet`).
   - **Automatic OMX Conversion**: Automatically converts `od.geojson` to binary `od.omx` on-the-fly if missing.

2. **Benchmark Verification (`sta test`)**:
   - Validates baseline assignment results (`scenario_00000.parquet`) against ground-truth TNTP benchmarks (`flows.geojson`).
   - Computes weighted MAPE (wMAPE), relative error pass rates ($\le 5\%$ and $\le 10\%$), and flow-weighted accuracy.

3. **OpenMatrix Utilities (`sta convert`)**:
   - Fast multi-processed conversion between GeoJSON tables (`od.geojson`) and OpenMatrix files (`od.omx`).

---

## Directory Structure

### Expected Input Structure (`data/<network>/scenarios_geojson/`)
The solver expects scenario directories produced by `scenario-generation`:

```text
code/
└── data/
    └── anaheim/
        └── scenarios_geojson/
            ├── scenario_00000/
            │   ├── nodes.geojson      # Node geometry & IDs
            │   ├── links.geojson      # Link topology, capacity, free-flow time, b, power
            │   └── od.geojson         # Demand table (auto-converted to od.omx if missing)
            ├── scenario_00001/
            └── ...
```

### Generated Output Structure (`data/<network>/scenarios_sta_results/`)
Results are exported as binary Parquet files:

```text
code/
└── data/
    └── anaheim/
        └── scenarios_sta_results/
            ├── scenario_00000/
            │   ├── scenario_00000.parquet            # Link flows, V/C, congested travel times
            │   └── scenario_00000_convergence.parquet # Iteration-wise relative gap log
            ├── scenario_00001/
            │   ├── scenario_00001.parquet
            │   └── scenario_00001_convergence.parquet
            └── ...
```

#### Output Parquet Schema (`<scenario_id>.parquet`):
- `link_id`: Unique link identifier
- `a_node`, `b_node`: Origin and destination node IDs of the physical link
- `capacity`: Link capacity
- `free_flow_time`: Uncongested travel time
- `flow`: Equilibrium PCE traffic volume ($\hat{f}_r$)
- `volume_capacity_ratio`: Volume-to-Capacity ratio ($V/C = \text{flow} / \text{capacity}$)
- `congested_time`: Congested travel time calculated via BPR function

---

## Installation & Environment Setup

Managed via [`pixi`](https://pixi.sh):

```bash
# Using Pixi (Recommended)
pixi run -e static-assignment <command>

# Or standalone editable install
pip install -e ./libs/static-assignment
```

---

## Command-Line Usage

### 1. Running Static Traffic Assignment (`run`)

```bash
# Run assignment for all Anaheim scenarios (using Pixi)
pixi run -e static-assignment run anaheim

# Overwrite existing results & use custom process count
pixi run -e static-assignment run chicago --overwrite -p 8

# Direct CLI usage
sta run anaheim --path data --overwrite -p 4
```

#### Options:
| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `NETWORK` | `TEXT` | *Required* | Network folder name under data path (e.g. `anaheim`, `chicago`). |
| `--path` | `TEXT` | `data` | Base directory containing scenario folders. |
| `-p`, `--processes` | `INT` | `CPU - 2` | Number of parallel processes to use. |
| `--overwrite` | `FLAG` | `False` | Overwrites existing `scenarios_sta_results` output directory. |

---

### 2. Validating Assignment Accuracy (`test`)

```bash
# Validate assignment against ground truth benchmark
pixi run -e static-assignment test anaheim

# Direct CLI usage
sta test anaheim --path data
```

---

### 3. OpenMatrix Conversion Utilities (`convert`)

```bash
# Convert all od.geojson to od.omx manually
sta convert to-omx anaheim

# Delete od.geojson after valid od.omx creation
sta convert cleanup anaheim

# Convert od.omx back to od.geojson
sta convert to-geojson anaheim
```
