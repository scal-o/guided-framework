import random
import copy
from pathlib import Path

import click
import mlflow
import torch
import torch_geometric as pg
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from ml_static.config import Config, ConfigOrigin, load_config_with_origin
from ml_static.context_manager import RunContext
from ml_static.data import DatasetSplit, InfiniteBatchSampler
from ml_static.early_stopping import EarlyStopping
from ml_static.factories import create_optimizer, create_scheduler
from ml_static.loss_curriculum import LossCurriculumScheduler
from ml_static.losses import LossWrapper
from ml_static.models import model_factory
from ml_static.tracker import MLflowtracker
from ml_static.training import freeze_parameters_from_config, run_epoch, run_test
from ml_static.utils import get_project_root, generate_run_name

# Add nvcc to environment PATH if CUDA_HOME is available
import sys
from torch.utils.cpp_extension import CUDA_HOME

# torch.compile (Inductor backend) relies on Triton, which is officially supported on Linux/WSL.
# On native Windows, PyTorch runs in high-performance eager mode with AMP (float16).
COMPILE_MODEL = False
if CUDA_HOME and sys.platform != "win32":
    import os

    os.environ["PATH"] = os.path.join(CUDA_HOME, "bin") + os.pathsep + os.environ["PATH"]
    COMPILE_MODEL = True


def run_training(
    config: Config,
    origin: ConfigOrigin,
    check_run: bool = False,
) -> tuple:
    """
    Execute GNN model training, validation, and testing.

    Args:
        config: Configuration object containing model, dataset, and training parameters.
        origin: Configuration origin metadata (paths to source YAML files).
        check_run: If True, runs on a single data sample to verify model convergence.

    Returns:
        Tuple of (trained_model, dataset_split)
    """

    # 1. Device Selection (CUDA GPU if available, else CPU)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Using device:", device)

    # 2. Model & Seed Resolution (Support for fresh training vs transfer learning fine-tuning)
    resolved_base_run_id: str | None = None
    if config.training.mode == "finetune":
        init_cfg = config.training.init

        run_id = init_cfg.run_id if init_cfg is not None else None
        download_path = (
            init_cfg.download_path if init_cfg is not None else Path("downloaded_models")
        )
        force_download = init_cfg.force_download if init_cfg is not None else False

        with RunContext(
            run_id=run_id,
            download_path=download_path,
            force=force_download,
        ) as ctx:
            resolved_base_run_id = ctx.run_id
            model = ctx.model.to(device)
            seed = ctx.seed
    else:
        model = model_factory(config).to(device)
        seed = (
            config.training.seed if config.training.seed is not None else random.randint(0, 10000)
        )

    # 3. Reproducibility & Dataset Split Instantiation
    dataset_split = DatasetSplit.from_config(config, seed)
    original_model = model

    if COMPILE_MODEL:
        try:
            print("Compiling model with torch.compile...")
            model = torch.compile(model, dynamic=True)
        except Exception as e:
            print(f"torch.compile unavailable ({e}). Falling back to PyTorch eager execution.")
            model = original_model

    pg.seed_everything(seed)

    # 4. Data Loaders & Batch Sampler
    print("Creating sampler and dataloader...")
    batch_sampler = InfiniteBatchSampler(
        dataset_split["train"],
        config.training.batch_size,
        seed=seed,
        shuffle=True,
        drop_last=True,
    )

    train_loader = DataLoader(
        dataset_split["train"],
        batch_sampler=batch_sampler,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        dataset_split["val"],
        batch_size=config.training.batch_size,
        pin_memory=True,
        shuffle=False,
    )

    # Cache validation loader in RAM for fast evaluation
    val_loader = list(val_loader)
    test_loader = DataLoader(
        dataset_split["test"],
        batch_size=config.training.batch_size,
        shuffle=False,
    )

    data_sample = next(iter(train_loader))

    if check_run:
        data_iterator = [data_sample]
        run_description = "Check run (overfitting test)"
    else:
        data_iterator = iter(train_loader)
        run_description = "Training"

    # 5. Loss Function, Transform Registration & Curriculum Scheduler
    loss = LossWrapper.from_config(config)

    transform = copy.deepcopy(dataset_split.transform).to(device)
    loss.register_transform(transform)

    if loss.mask_type is not None and loss.mask_evaluation == "unobserved":
        batch_sampler = InfiniteBatchSampler(
            copy.deepcopy(dataset_split["train"]),
            64,
            seed=seed,
            shuffle=False,
            drop_last=False,
        )
        val_loader = DataLoader(
            copy.deepcopy(dataset_split["train"]),
            batch_sampler=batch_sampler,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=4,
        )
        val_loader = iter(val_loader)

    curriculum = LossCurriculumScheduler.from_config(config.loss.curriculum)

    # 6. Parameter Freezing (for fine-tuning / transfer learning)
    freeze_report = None
    if config.training.mode == "finetune" and config.training.freeze is not None:
        freeze_report = freeze_parameters_from_config(model, config.training.freeze)

    # 7. Optimizer & Learning Rate Scheduler
    trainable_params = (p for p in model.parameters() if p.requires_grad)
    optimizer = create_optimizer(config.optimizer, trainable_params)

    scheduler_runtime_params = {
        "epochs": config.training.epochs,
        "steps_per_epoch": len(train_loader),
    }
    scheduler = create_scheduler(config.scheduler, optimizer, scheduler_runtime_params)

    # 8. Early Stopping Setup
    early_stopping = None
    if config.early_stopping.enabled:
        early_stopping = EarlyStopping(
            patience=config.early_stopping.patience,
            min_delta=config.early_stopping.min_delta,
            mode=config.early_stopping.mode,
            warmup_epochs=config.early_stopping.warmup_epochs,
        )

    epochs = config.training.epochs

    # 9. MLflow Experiment Tracking & Logging
    tracker = MLflowtracker()
    run_name = generate_run_name(config)
    print("Run name: ", run_name)

    with mlflow.start_run(run_name=run_name):
        tracker.log_effective_configs(origin)
        tracker.log_params(config.raw_config)
        tracker.log_params(config.raw_model)
        tracker.log_seed(seed)
        tracker.log_dataset_info(dataset_split)
        tracker.log_training_mode(config.training.mode)

        if config.training.mode == "finetune" and resolved_base_run_id is not None:
            tracker.log_finetune_base_run(resolved_base_run_id)

        if freeze_report is not None:
            tracker.log_finetune_freeze_report(freeze_report)

        # 10. Training Loop
        for epoch in tqdm(range(1, epochs + 1), desc=run_description):
            # Apply physics-informed loss curriculum updates (e.g. conservation loss ramp)
            update = curriculum.weights_for_epoch(epoch)
            if update.weights:
                loss.set_weights(**update.weights)
                tracker.log_loss_weight_updates(epoch, update.weights)

            # Execute training & validation steps
            e_train_loss, e_val_loss, train_components, val_components = run_epoch(
                model,
                data_iterator,
                val_loader,
                optimizer,
                loss,
                device,
            )

            # Step learning rate scheduler
            if config.scheduler.warmup_epochs > epoch:
                pass
            else:
                if check_run:
                    scheduler.step(e_train_loss)
                else:
                    scheduler.step(e_val_loss)

            current_lr = scheduler.get_last_lr()[0]

            # Log metrics to MLflow
            tracker.log_epoch(
                epoch,
                e_train_loss,
                e_val_loss,
                train_components,
                val_components,
                current_lr,
            )

            # Check early stopping conditions
            if early_stopping is not None:
                if early_stopping(e_val_loss, original_model, epoch):
                    print(
                        f"\nEarly stopping triggered at epoch {epoch}. "
                        f"Best validation loss: {early_stopping.best_metric:.4f} at epoch {early_stopping.best_epoch}"
                    )
                    break

            if epoch % 10 == 0:
                print(
                    f"Epoch {epoch}/{epochs} - Train Loss: {e_train_loss:.4f} - Val Loss: {e_val_loss:.4f} - LR: {current_lr:.2e}"
                )

                if val_components:
                    comp_str = "Loss components: " + " - ".join(
                        f"{k.replace('unweighted_', '')}: {v:.2f}"
                        for k, v in val_components.items()
                        if k.startswith("unweighted_")
                    )
                    print(comp_str)

        # Restore best model checkpoint if early stopping was active
        if early_stopping is not None:
            early_stopping.load_best_model(original_model)
            tracker.log_best_model_info(early_stopping.best_epoch, early_stopping.best_metric)
            print(
                f"\nLoaded best original_model from epoch {early_stopping.best_epoch} "
                f"(val_loss: {early_stopping.best_metric:.4f})"
            )

        # 11. Final Evaluation & Report Generation
        test_loss = run_test(original_model, test_loader, loss, device)
        tracker.log_test_loss(test_loss)
        print(f"Test Loss: {test_loss:.4f}")

        tracker.log_model(original_model, config.model.type, dataset_split["train"][0])
        tracker.log_training_curves()

        print("--- Computing Performance Statistics ---")
        datasets = {
            "train": dataset_split["train"],
            "validation": dataset_split["val"],
            "test": dataset_split["test"],
        }
        stats = tracker.log_all_performance_reports(original_model, datasets)
        print(stats)

        print("--- Logging Sample Scenario Predictions ---")
        tracker.log_random_scenario_predictions(original_model, datasets, num_scenarios=5)

        return original_model, dataset_split


@click.command("train")
@click.option(
    "-c",
    "--config-path",
    default=None,
    help="Path to YAML configuration file. Defaults to conf_run.yaml.",
)
@click.option(
    "--check-run",
    is_flag=True,
    default=False,
    help="Run a check run on a single data sample to verify model convergence.",
)
def train_model(
    config_path: Path | str | None = None,
    check_run: bool = False,
) -> tuple:
    """
    Train a GNN model on static traffic assignment data.

    Returns:
        Tuple of (model, dataset_split)
    """
    print("--- Training GNN Model ---")

    if config_path is None:
        config_path = get_project_root() / "configs" / "conf_run.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    config, origin = load_config_with_origin(config_path)
    print(f"Configuration loaded from {origin.run_config_path}")
    print(f"Dataset: {config.dataset.full_path}")
    print(f"Check run: {check_run}")

    try:
        model, dataset_split = run_training(config, origin, check_run=check_run)
        print("--- Training Complete ---")
        return model, dataset_split
    except Exception as e:
        raise Exception(f"Training failed. An unexpected error occurred: {e}") from e


if __name__ == "__main__":
    train_model()
