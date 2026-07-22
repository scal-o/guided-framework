# `ml-static`

A PyTorch & PyTorch Geometric package for training, evaluating, and transferring Heterogeneous Graph Attention Networks (HetGAT) equipped with the **GUIDED** network-agnostic feature initialization layer.

This package implements the machine learning models, loss functions, and experimental benchmarks evaluated in **"GUIDED Network-Agnostic Feature Initialization for Spatial Transferability in GNN-based Models"** (Scalese et al., 2026).

---

## Key Features

1. **GUIDED-HetGAT Architecture (`models/`)**:
   - **GUIDED Feature Initialization**: Projects scalar travel demand $d_q$ on virtual links into 32-dim edge embeddings $\mathbf{e}_q$ (via linear MLP or Gaussian RBF expansion), followed by directional scatter-sum aggregation into initial physical node embeddings $\mathbf{x}_v^{(0)} = [\mathbf{x}_{\text{out},v} \parallel \mathbf{x}_{\text{in},v}] \in \mathbb{R}^{64}$.
   - **Heterogeneous Graph Encoders**: Stack of Virtual Encoder (V-Encoder) and Real Encoder (R-Encoder) attention layers.
   - **Edge Flow Predictor**: Dedicated MLP predicting link Volume-to-Capacity Ratio ($V/C$).

2. **Physics-Informed Loss Function (`losses.py`)**:
   - Composite loss $\mathcal{L} = \lambda_v \mathcal{L}_v + \lambda_f \mathcal{L}_f + \lambda_c \mathcal{L}_c$:
     - $\mathcal{L}_v$: Supervised V/C prediction error (MAE).
     - $\mathcal{L}_f$: Supervised link flow error (PCE).
     - $\mathcal{L}_c$: Physics-based node flow conservation penalty.
   - **Curriculum Learning (`loss_curriculum.py`)**: Sigmoid ramp schedule introducing conservation penalty $\mathcal{L}_c$ after initial V/C convergence.

3. **Experiment Suite Automation (`batch_runner.py`)**:
   - Automated sequential execution of experiment suites matching paper evaluation phases A through D:
     - **Experiment A**: Intra-network generalization ceiling.
     - **Experiment B**: Sample efficiency under data scarcity (200 scenarios).
     - **Experiment C**: Semi-supervised inference under partial link observability (50% flow masking).
     - **Experiment D**: Inter-network transferability & layer-wise ablation fine-tuning.

---

## Configuration Files (`configs/`)

| Config File | Description |
| :--- | :--- |
| `conf_run.yaml` | Master template for single training runs. |
| `conf_finetune.yaml` | Master template for transfer learning / fine-tuning experiments (Exp D). |
| `conf_mlflow.yaml` | MLflow experiment tracking URI and experiment name configuration. |
| `conf_model_linear_init.yaml` | GUIDED Linear variant (`GUIDED-HetGAT-lin`). |
| `conf_model_rbf_init.yaml` | GUIDED RBF variant (`GUIDED-HetGAT-rbf`). |
| `conf_model_preproc_padded.yaml` | Transductive baseline HetGAT model (zero-padded to 950 nodes). |
| `experiment_suite_A.yaml` | Experiment A suite definition. |
| `experiment_suite_B.yaml` | Experiment B suite definition. |
| `experiment_suite_C.yaml` | Experiment C suite definition. |
| `experiment_suite_D.yaml` | Experiment D suite definition. |

---

## MLflow Experiment Tracking & Artifacts

By default, training runs use **local file-based MLflow tracking** configured in `configs/conf_mlflow.yaml`:

```yaml
tracking_uri: ./mlruns
experiment: "GUIDED-HetGAT"
```

- **Local Storage (`./mlruns/`)**: Does not require any external tracking server or Databricks setup. All metrics, parameters, and model artifacts are saved directly on disk.
- **Logged Artifacts**:
  - Model Checkpoints (`model/model.pt`)
  - Effective YAML configurations (`configs_effective/`)
  - Evaluation summary reports (`stats.txt`, `performance_reports.json`)
  - Loss & metric training curves (`training_curves.png`)
- **Launching MLflow UI**: To view training curves and compare runs in your browser, launch:
  ```bash
  pixi run -e ml-static mlflow ui
  ```
  Then open `http://127.0.0.1:5000` in your web browser.
- **Using a Remote/Custom Server**: If you run a central MLflow tracking server, update `tracking_uri` in `configs/conf_mlflow.yaml` (e.g., `tracking_uri: http://localhost:5000` or `http://your-server-ip:5000`).

---

## Installation & Environment Setup

Managed via [`pixi`](https://pixi.sh):

```bash
# Using Pixi (Recommended)
pixi run -e ml-static <command>

# Standalone editable install
pip install -e ./libs/ml-static
```

---

## Command-Line Usage

### 1. Run Single Model Training (`train`)

```bash
# Run single training run using Pixi
pixi run -e ml-static train -c configs/conf_run.yaml

# Override model or dataset parameters
train -c configs/conf_run.yaml --model.name linear_init --dataset.name anaheim_a
```

### 2. Run Batch Experiment Suites (`train-suite`)

```bash
# Run Experiment A suite (Intra-network generalization)
pixi run -e ml-static train-suite -s configs/experiment_suite_A.yaml

# Run Experiment B suite (Data scarcity)
pixi run -e ml-static train-suite -s configs/experiment_suite_B.yaml

# Run Experiment C suite (Semi-supervised inference)
pixi run -e ml-static train-suite -s configs/experiment_suite_C.yaml

# Run Experiment D suite (Inter-network transfer learning)
pixi run -e ml-static train-suite -s configs/experiment_suite_D.yaml
```

---

## Evaluation Metrics

During training and testing, models are evaluated on both regression and transportation domain metrics:

- **MAE (Mean Absolute Error)**: Average absolute flow prediction error across physical links.
- **RMSN (Normalized Root Mean Square Error)**: Scale-normalized prediction error.
- **$R^2$ (Coefficient of Determination)**: Overall goodness-of-fit.
- **% GEH < 5**: Percentage of links satisfying standard engineering acceptability criteria ($\text{GEH} < 5$).
- **FCN (Flow Conservation Error)**: Relative node inflow-outflow imbalance across the network.
