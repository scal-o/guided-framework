from __future__ import annotations

import importlib.util
import shutil
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Self

import mlflow
import torch
import yaml
from mlflow.entities import ViewType

from ml_static.utils import get_project_root

# lock to prevent concurrent sys.path/sys.modules manipulation
_import_lock = threading.RLock()

# MLflow tag used to indicate that a run is a fine-tune run and points to the base run
# that defines the model code and architecture.
_FINETUNE_BASE_RUN_TAG = "finetune.base_run_id"


def get_tracking_config() -> dict[str, str]:
    """
    Load MLflow tracking configuration from conf_mlflow.yaml.

    Returns:
        Dictionary with 'tracking_uri' and 'experiment' keys.
    """
    config_path = get_project_root() / "configs" / "conf_mlflow.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"MLflow configuration file not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    tracking_uri = config.get("tracking_uri")
    experiment_name = config.get("experiment")

    if not tracking_uri or not experiment_name:
        raise ValueError(f"'tracking_uri' and 'experiment' must be specified in {config_path}")

    return config


def list_runs(
    config: dict[str, str] | None = None,
    filter_string: str | None = None,
) -> dict[str, str]:
    """
    List all runs for an experiment with their key metrics and parameters.

    Args:
        config: MLflow tracking configuration dictionary. If None, loads from conf_mlflow.yaml.
        filter_string: MLflow filter string (e.g., "metrics.test_loss < 0.1").

    Returns:
        Dictionary mapping run IDs to run names.
    """
    # load defaults from config if not provided
    if not config:
        config = get_tracking_config()

    tracking_uri = config["tracking_uri"]
    experiment_name = config["experiment"]

    # set tracking URI
    mlflow.set_tracking_uri(tracking_uri)

    # get experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found")

    # search runs
    runs = mlflow.search_runs(  # type: ignore
        experiment_ids=[experiment.experiment_id],
        filter_string=filter_string if filter_string else "",
        run_view_type=ViewType.ACTIVE_ONLY,
    )

    if runs.empty:  # type: ignore
        print(f"No runs found for experiment '{experiment_name}'")
        return dict()

    # clean up the dataframe for better readability
    # keep only the most relevant columns
    relevant_cols = ["run_id", "tags.mlflow.runName", "start_time"]
    runs = runs[relevant_cols]  # type: ignore
    runs = runs.sort_values("start_time", ascending=False)  # type: ignore

    runs = {name: id for id, name in zip(runs["run_id"], runs["tags.mlflow.runName"])}  # type: ignore

    return runs


def resolve_run_id(run_config: dict[str, str], run_id: str | None = None) -> str:
    runs = list_runs(run_config)

    # if run id is none, default to last run
    if run_id is None:
        return list(runs.values())[0]

    # check if the given run id is in the keys / values of the dict
    if run_id in runs.keys():
        resolved = runs.get(run_id)
        if resolved is None:
            raise KeyError(
                f"Given run name {run_id} resolved to None in mlflow experiment {run_config['experiment']}."
            )
        return resolved
    elif run_id in runs.values():
        return run_id
    else:
        raise KeyError(
            f"Given run id {run_id} could not be found in mlflow experiment {run_config['experiment']}."
        )


class RunContext:
    """Context manager for loading and managing model artifacts from MLflow runs.

    This class is "smart" about fine-tuning:
    - If the requested run has tag `finetune.base_run_id`, the base run provides the code/config
      (for model reconstruction) and the requested run provides the fine-tuned weights.
    - Otherwise, the requested run is treated as self-contained (code/config/weights all come
      from the same run).
    """

    def __init__(
        self,
        run_id: str | None = None,
        config: dict[str, str] | None = None,
        download_path: Path | str = "downloaded_models",
        force: bool = False,
    ):
        """
        Initialize the context for a run.

        Args:
            run_id: MLflow run ID. Defaults to last run if None. -> Can also be the name of the run.
            download_path: Base path for downloaded models.
            tracking_uri: MLflow tracking URI (defaults to conf_mlflow.yaml).
            force: If True, re-download even if already exists.
            dataset_config_source: Which run's config to use for dataset reconstruction.
                - "base": Use the base run config (default; recommended).
                - "weights": Use the weights/fine-tune run config (only supported if that run
                  logged effective configs).
        """

        # if no config provided, load from file
        self.run_config = get_tracking_config() if config is None else config

        # Keep the caller-provided run id for logging/debugging; may be None.
        self.requested_run_id: str | None = run_id

        # Resolve the actual run id we'll use (defaults to last run if not provided).
        self.run_id: str = resolve_run_id(self.run_config, self.requested_run_id)

        # set tracking uri
        self.tracking_uri = self.run_config["tracking_uri"]

        # set download path for artifacts
        self.download_path = Path(download_path)

        self.force: bool = force

        # Determine lineage:
        # - base_run_id is where we load code/config to reconstruct the model
        # - weights_run_id is where we load the model checkpoint weights
        mlflow.set_tracking_uri(self.tracking_uri)
        weights_run = mlflow.get_run(self.run_id)
        base_run_id = weights_run.data.tags.get(_FINETUNE_BASE_RUN_TAG)

        if base_run_id:
            # Validate base run exists early for clearer errors
            _ = mlflow.get_run(base_run_id)
            self.base_run_id: str = base_run_id
            self.weights_run_id: str = self.run_id
            self.is_finetune: bool = True
        else:
            self.base_run_id = self.run_id
            self.weights_run_id = self.run_id
            self.is_finetune = False

        # create model directory paths for base + weights
        self.base_model_dir: Path = self._resolve_model_dir(self.base_run_id)
        self.weights_model_dir: Path = self._resolve_model_dir(self.weights_run_id)

        # Prefix used for any dynamic module naming (currently informational)
        self.module_prefix: str = f"mlstatic_base_{self.base_run_id}_weights_{self.weights_run_id}"

        self.models_module: Any = None
        self.data_module: Any = None
        self.config_module: Any = None

        self._model: Any = None
        self._config: Any = None
        self._dataset: Any = None
        self._data_split: Any = None

    def __enter__(self) -> Self:
        """Download artifacts and surgically load modules."""
        self._download_artifacts()

        # The core logic is now much safer and cleaner
        with _import_lock:
            print("starting module load")
            # 1. Save a reference to any currently installed ml_static modules
            self._saved_modules = {}
            for name in list(sys.modules.keys()):
                if name == "ml_static" or name.startswith("ml_static."):
                    self._saved_modules[name] = sys.modules.pop(name)

            try:
                # 2. Define the path to the downloaded package
                # For fine-tuning, we always import the BASE run's code for model reconstruction.
                code_path = self.base_model_dir / "code"
                package_init_path = code_path / "__init__.py"

                if not package_init_path.is_file():
                    raise FileNotFoundError(
                        f"The package init file '__init__.py' was not found in {code_path}"
                    )

                # 3. Create a module spec for the downloaded package.
                # This tells the import system that the 'code' directory is the 'ml_static' package.
                spec = importlib.util.spec_from_file_location("ml_static", package_init_path)

                if spec is None or spec.loader is None:
                    raise ImportError(f"Could not create a module spec for {package_init_path}")

                # 4. Create the module and add it to sys.modules.
                ml_static_module = importlib.util.module_from_spec(spec)
                sys.modules["ml_static"] = ml_static_module

                # Execute the module code (the __init__.py)
                spec.loader.exec_module(ml_static_module)

                # 5. Now, absolute imports work correctly.
                import ml_static.config as config
                import ml_static.data as data
                import ml_static.models as models

                self.config_module = config
                self.data_module = data
                self.models_module = models

            except Exception:
                # If anything goes wrong, clean up immediately and restore state
                self.__exit__(*sys.exc_info())
                raise
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Clean up modules and restore the original environment."""
        with _import_lock:
            # 1. Remove all modules that were loaded from the downloaded code
            loaded_code_dir = str(self.base_model_dir / "code")
            for name, module in list(sys.modules.items()):
                # A module is one of ours if it has a __file__ pointing to our code dir
                if (
                    hasattr(module, "__file__")
                    and module.__file__
                    and str(module.__file__).startswith(loaded_code_dir)
                ):
                    del sys.modules[name]

            # 2. Restore the original modules that were saved in __enter__
            if hasattr(self, "_saved_modules"):
                sys.modules.update(self._saved_modules)
                del self._saved_modules  # Clean up the saved dict to avoid holding refs

        self._cleanup_data()

        if exc_type is not None:
            print(f"Error occurred: {exc_type.__name__}: {exc_val}")

    def _resolve_model_dir(self, run_id: str) -> Path:
        """Compute the on-disk artifact directory for a given MLflow run."""
        mlflow.set_tracking_uri(self.tracking_uri)
        run = mlflow.get_run(run_id)
        experiment = mlflow.get_experiment(run.info.experiment_id)
        run_name = run.data.tags.get("mlflow.runName", run_id)
        return self.download_path / experiment.name.lstrip("/") / run_name

    def _download_artifacts(self) -> None:
        """
        Download necessary artifacts for both base and weights runs.

        - Base run: full artifact download (needs code + configs)
        - Weights run: full artifact download (simple and robust; could be optimized later)
        """
        # BASE run
        if self.base_model_dir.exists():
            if self.force:
                shutil.rmtree(self.base_model_dir)
            else:
                print(f"Base run already downloaded at: {self.base_model_dir}")
        if not self.base_model_dir.exists():
            self.base_model_dir.parent.mkdir(parents=True, exist_ok=True)
            print(f"Downloading artifacts for base run {self.base_run_id}...")
            mlflow.artifacts.download_artifacts(  # type: ignore
                run_id=self.base_run_id, artifact_path="", dst_path=str(self.base_model_dir)
            )
            print(f"Base run downloaded successfully to: {self.base_model_dir}")

        # WEIGHTS run (may equal base)
        if self.weights_run_id == self.base_run_id:
            return

        if self.weights_model_dir.exists():
            if self.force:
                shutil.rmtree(self.weights_model_dir)
            else:
                print(f"Weights run already downloaded at: {self.weights_model_dir}")
                return

        self.weights_model_dir.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading artifacts for weights run {self.weights_run_id}...")
        mlflow.artifacts.download_artifacts(  # type: ignore
            run_id=self.weights_run_id, artifact_path="", dst_path=str(self.weights_model_dir)
        )
        print(f"Weights run downloaded successfully to: {self.weights_model_dir}")

    def _cleanup_data(self) -> None:
        """
        Remove all data instances from class.
        """

        self._model = None
        self._config = None
        self._dataset = None

    def _get_logged_seed(self, run_id: str) -> int | None:
        """Fetch the training seed logged to MLflow for a given run.

        Args:
            run_id: MLflow run id to read params from.

        Returns:
            The seed as an int if present and parseable, otherwise None.
        """
        try:
            run = mlflow.get_run(run_id)
        except Exception:
            return None

        raw = (run.data.params or {}).get("seed")
        if raw is None:
            return None

        try:
            # MLflow params are stored as strings
            return int(raw)
        except (TypeError, ValueError):
            return None

    @property
    def seed(self) -> int:
        """
        Retrieve the correct seed logged to MLflow for the selected data source run.
        """
        seed = getattr(self.base_config.training, "seed", None)
        if isinstance(seed, int):
            return seed

        logged = self._get_logged_seed(self.run_id)
        if logged is not None:
            return logged

        raise ValueError(
            "Could not determine seed: base_config.training.seed is not an int and "
            f"MLflow run '{self.run_id}' did not log a parseable 'seed' param."
        )

    def _load_config_from_dir(self, cfg_root: Path) -> Any:
        """
        Load a Config instance from the given artifact root.

        Prefers effective configs (reproducible single source of truth) and falls back
        to legacy configs.

        Args:
            cfg_root: Directory containing downloaded artifacts for a run.

        Returns:
            Loaded Config instance.
        """
        effective_dir = cfg_root / "configs_effective"
        legacy_dir = cfg_root / "configs"

        effective_run_path = effective_dir / "conf_run_effective.yaml"
        legacy_run_path = legacy_dir / "conf_run.yaml"

        if effective_run_path.exists():
            run_config_path = effective_run_path

            with open(run_config_path) as f:
                run_raw = yaml.safe_load(f) or {}

            model_name = (run_raw.get("model") or {}).get("name", "HetGAT")
            model_config_path = effective_dir / f"conf_model_{model_name}_effective.yaml"

            if not model_config_path.exists():
                raise FileNotFoundError(
                    f"Effective model config not found: {model_config_path}. "
                    f"Expected based on model.name='{model_name}' in {run_config_path}"
                )

            return self.config_module.ConfigLoader.from_yaml(
                run_config_path,
                model_config_path=Path(model_config_path),
            )

        if legacy_run_path.exists():
            run_config_path = legacy_run_path

            with open(run_config_path) as f:
                run_raw = yaml.safe_load(f) or {}

            model_name = (run_raw.get("model") or {}).get("name", "HetGAT")
            model_config_path = legacy_dir / f"conf_model_{model_name}.yaml"

            if not model_config_path.exists():
                return self.config_module.load_config(run_config_path)

            return self.config_module.ConfigLoader.from_yaml(
                run_config_path,
                model_config_path=Path(model_config_path),
            )

        raise FileNotFoundError(
            "No configuration found in run artifacts. "
            f"Looked for effective: {effective_run_path} and legacy: {legacy_run_path}"
        )

    @property
    def base_config(self) -> Any:
        """
        Build the Config instance from the BASE run (architecture/transforms source of truth).

        Returns:
            Config instance from the base run's configuration.
        """
        if not self._config:
            print("--- Loading base configuration")
            self._config = self._load_config_from_dir(self.base_model_dir)

        return self._config

    @property
    def weights_config(self) -> Any:
        """
        Build the Config instance from the WEIGHTS run.

        For fine-tuned runs, this attempts to load the fine-tune run's config (only works
        if that run logged configs). If unavailable, it falls back to `base_config`.

        Returns:
            Config instance from the weights run's configuration, or `base_config` fallback.
        """
        if not self.is_finetune:
            return self.base_config

        if not hasattr(self, "_weights_config"):
            self._weights_config = None

        if self._weights_config is not None:
            return self._weights_config

        try:
            print("--- Loading weights configuration")
            self._weights_config = self._load_config_from_dir(self.weights_model_dir)
        except FileNotFoundError:
            self._weights_config = self.base_config

        return self._weights_config

    @property
    def config(self) -> Any:
        """
        Backwards-compatible alias for the base run configuration.
        """
        return self.base_config

    @property
    def model(self) -> torch.nn.Module:
        """
        Build the model using the downloaded artifacts.

        Returns:
            The loaded (cpu) PyTorch model ready for inference.

        Notes:
            For fine-tuned runs, the model is instantiated from the BASE run config/code,
            and weights are loaded from the WEIGHTS run checkpoint.
        """

        if not self._model:
            print("--- Loading model")

            # load checkpoint state from the weights run
            checkpoint_path = self.weights_model_dir / "model" / "model.pt"
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")

            checkpoint = torch.load(checkpoint_path, weights_only=False)
            state_dict = checkpoint.get("state_dict")
            if state_dict is None:
                raise ValueError(
                    f"Checkpoint at {checkpoint_path} does not contain 'state_dict' field"
                )

            # verify factory exists
            if not hasattr(self.models_module, "model_factory"):
                raise AttributeError("Models module does not define 'model_factory'")

            # create model from BASE config and load WEIGHTS state dict
            self._model = self.models_module.model_factory(self.base_config)
            self._model.load_state_dict(state_dict, strict=True)

        return self._model

    @property
    def dataset(self) -> Any:
        """
        Build a dataset using the correct STADataset class from the run.

        Returns:
            STADataset instance compatible with the model.
        """

        if not self._dataset:
            print("--- Loading dataset")

            # verify STADataset exists
            if not hasattr(self.data_module, "STADataset"):
                raise AttributeError("Data module does not define 'STADataset' class")

            self._dataset = self.data_module.STADataset.from_config(
                self.weights_config, force_reload=True
            )

        return self._dataset

    @property
    def data_split(self) -> Any:
        """
        Create DatasetSplit using the seed from the run configuration.
        Splits are deterministically regenerated from the seed.

        Returns:
            DatasetSplit instance with train/val/test splits.

        Notes:
            The TRAINING seed used for the split is taken from the loaded config.
            For runs where the seed was randomized at runtime and not written into the
            YAML, exact split recreation is not possible from config alone.
        """

        if not self._data_split:
            print("--- Creating data split from config")

            # verify DatasetSplit exists
            if not hasattr(self.data_module, "DatasetSplit"):
                raise AttributeError("Data module does not define 'DatasetSplit' class")

            # This uses `config.training.seed` (or falls back to 42 inside DatasetSplit).
            self._data_split = self.data_module.DatasetSplit.from_config(
                self.weights_config, force_reload=False, seed=self.seed
            )

        return self._data_split
