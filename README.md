# GUIDED Network-Agnostic Feature Initialization for Spatial Transferability in GNN-based Models

Official PyTorch implementation of the paper:
[**"GUIDED Network-Agnostic Feature Initialization for Spatial Transferability in GNN-based Models"**](https://arxiv.org/abs/2607.19270) (Scalese et al., 2026).

[![arXiv](https://img.shields.io/badge/arXiv-2607.19270-b31b1b.svg)](https://arxiv.org/abs/2607.19270)

---

## Overview

Graph Neural Networks (GNNs) have emerged as fast, data-driven surrogates for macroscopic Static Traffic Assignment (STA). However, standard GNN models rely on transductive feature initializations that tie travel demand to fixed network topologies, preventing seamless transfer to new urban environments.

To overcome this structural limitation, this research proposes a network-agnostic initialization framework, termed **Geometrically Unconstrained Inductive Demand EmbeDding (GUIDED)**. By injecting travel demand as a scalar attribute on auxiliary virtual links rather than as specific node features, GUIDED standardizes the input space regardless of network scale or the number of active origin-destination pairs.

---

## Repository Structure

The project is organized as a modular workspace containing three standalone Python packages under `libs/`:

```text
./
├── libs/
│   ├── scenario-generation/     # TNTP ingestion & stochastic demand/capacity scenario generator
│   ├── static-assignment/       # AequilibraE User Equilibrium (UE) traffic assignment solver
│   └── ml-static/               # GUIDED-HetGAT models, physics losses, and experiment suites
├── networks/                    # TNTP network topologies (Anaheim, Chicago)
├── data/                        # Generated scenario datasets (Dataset A & B)
├── pixi.toml                    # Global environment & task manager configuration
└── pixi.lock                    # Locked reproducible environment dependencies
```

### Modular Packages:
- **[`libs/scenario-generation`](libs/scenario-generation/README.md)**: Converts raw TNTP files into master GeoPackages and generates perturbed scenarios for **Dataset A** (Uniform Scaling) and **Dataset B** (Dirichlet Distribution Shift).
- **[`libs/static-assignment`](libs/static-assignment/README.md)**: Solves deterministic User Equilibrium (Bi-Conjugate Frank-Wolfe, rgap $\le 10^{-5}$) using AequilibraE to compute ground-truth equilibrium flows.
- **[`libs/ml-static`](libs/ml-static/README.md)**: Implements PyTorch Geometric models, training pipelines, physics-informed losses, and automated batch experiment suites ($A, B, C, D$).

---

## Environment Setup

Environment and dependency management is handled via [`pixi`](https://pixi.sh):

```bash
# Install locked dependencies and verify environment
pixi info
```

---

## Quickstart Pipeline

### 1. Ingest Networks & Generate Scenarios (`scenario-generation`)

```bash
# 1. Compile TNTP files into master GeoPackage
pixi run -e scenario-generation initialize anaheim
pixi run -e scenario-generation initialize chicago

# 2. Generate Dataset A (Uniform Scaling, 700 scenarios)
pixi run -e scenario-generation generate anaheim -n 700 --dataset-type dataset_a --multiprocess
pixi run -e scenario-generation generate chicago -n 700 --dataset-type dataset_a --multiprocess

# 3. Generate Dataset B (Dirichlet Distribution Shift, 700 scenarios)
pixi run -e scenario-generation generate anaheim -n 700 --dataset-type dataset_b --multiprocess
pixi run -e scenario-generation generate chicago -n 700 --dataset-type dataset_b --multiprocess
```

### 2. Solve Static Traffic Assignment (`static-assignment`)

```bash
# Run User Equilibrium assignment for all generated scenarios
pixi run -e static-assignment run anaheim_a --overwrite
pixi run -e static-assignment run anaheim_b --overwrite
pixi run -e static-assignment run chicago_a --overwrite
pixi run -e static-assignment run chicago_b --overwrite

# Verify convergence against ground-truth TNTP benchmarks
pixi run -e static-assignment test anaheim_a
```

### 3. Model Training & Paper Experiment Suites (`ml-static`)

```bash
# Quick sanity check run on mini-batch (verify GPU CUDA setup)
pixi run -e ml-static python -m ml_static.run --check-run

# Run Experiment A Suite (Intra-network generalization ceiling)
pixi run -e ml-static train-suite -s configs/experiment_suite_A.yaml

# Run Experiment B Suite (Sample efficiency under data scarcity - 200 scenarios)
pixi run -e ml-static train-suite -s configs/experiment_suite_B.yaml

# Run Experiment C Suite (Semi-supervised inference under 50% link masking)
pixi run -e ml-static train-suite -s configs/experiment_suite_C.yaml

# Run Experiment D Suite (Inter-network transferability & layer sensitivity fine-tuning)
pixi run -e ml-static train-suite -s configs/experiment_suite_D.yaml
```

### 4. MLflow Experiment Tracking & Visualization

All training metrics, effective YAML configurations, and model checkpoints are saved locally in `./mlruns/`. To inspect training curves and compare model runs:

```bash
# Launch MLflow UI locally
pixi run -e ml-static mlflow ui
```
Then open `http://127.0.0.1:5000` in your web browser.

---

## Citation

If you use this codebase or the GUIDED framework in your research, please cite our paper:

```bibtex
@article{scalese2026guided,
  author = {Scalese, Alessandro and Narayanan, Santhanakrishnan and Antoniou, Constantinos},
  title = {GUIDED Network-Agnostic Feature Initialization for Spatial Transferability in GNN-based Models},
  journal = {arXiv preprint arXiv:2607.19270},
  year = {2026},
  eprint = {2607.19270},
  archiveprefix = {arXiv}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
