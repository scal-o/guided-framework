"""
Batch experiment runner for sequential training runs.

This runner supports applying overrides in two distinct scopes:
- Run config overrides (`updates.config`): Applied to the baseline run config template (`conf_run.yaml` or `conf_finetune.yaml`).
- Model config overrides (`updates.model`): Applied to the specific model architecture config (`conf_model_<model_name>.yaml`).

For each experiment in a suite, this module generates isolated, reproducible *effective* YAML files under:
`configs/effective/<experiment_name>/`

These effective YAML files are logged to MLflow by the training pipeline as a single source of truth for full run auditability.

Override Guardrails:
- Overrides are specified using dotted notation (e.g. `training.epochs: 50`).
- By default, overrides are STRICT: every intermediate parent key must exist and be a dictionary.
  This prevents typos from creating silent, ignored keys.
"""

from __future__ import annotations

from pathlib import Path

import click
import yaml

from ml_static.config import load_config_with_origin
from ml_static.run import run_training
from ml_static.utils import get_project_root


def _sanitize_experiment_name(name: str) -> str:
    """
    Sanitize experiment names to create safe filesystem folder names.
    """
    safe = "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in name.strip())
    return safe or "experiment"


class ConfigManager:
    """
    Manages loading baseline configuration templates and applying experiment-specific overrides
    to produce effective YAML files for each experiment in a suite.
    """

    def __init__(self, config_path: Path | None = None):
        """
        Initialize ConfigManager with a default baseline configuration path.

        Args:
            config_path: Path to baseline run configuration. Defaults to configs/conf_run.yaml.
        """
        if config_path is None:
            self.config_path = get_project_root() / "configs" / "conf_run.yaml"
        else:
            self.config_path = Path(config_path)

        self.configs_dir = self.config_path.parent
        self.effective_root = self.configs_dir / "effective"
        self.effective_root.mkdir(parents=True, exist_ok=True)

    def _resolve_template_config_path(self, template: str | None) -> Path:
        """
        Resolve the baseline run-config YAML path based on the specified template name.

        Supported Templates:
            - None or "run"  -> configs/conf_run.yaml (Standard training)
            - "finetune"    -> configs/conf_finetune.yaml (Fine-tuning / transfer learning)
        """
        if template is None:
            return self.config_path

        template_norm = template.strip().lower()
        if template_norm == "run":
            return self.configs_dir / "conf_run.yaml"
        if template_norm == "finetune":
            return self.configs_dir / "conf_finetune.yaml"

        raise ValueError(
            f"Unknown experiment template '{template}'. Supported options are 'run' or 'finetune'."
        )

    def load_run_dict(self, template: str | None = None) -> dict:
        """
        Load the baseline run configuration YAML into a Python dictionary.
        """
        config_path = self._resolve_template_config_path(template)
        if not config_path.exists():
            raise FileNotFoundError(
                f"Baseline run config not found: {config_path}. "
                f"Template='{template or 'run'}'."
            )

        with open(config_path) as f:
            data = yaml.safe_load(f)
        return data or {}

    def resolve_model_config_path(self, run_dict: dict) -> Path:
        """
        Resolve the model configuration YAML path from `model.name` in the run config dictionary.
        """
        model_name = (run_dict.get("model") or {}).get("name", "HetGAT")
        return self.configs_dir / f"conf_model_{model_name}.yaml"

    def load_model_dict(self, model_path: Path) -> dict:
        """
        Load a model configuration YAML into a Python dictionary.
        """
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model config file not found: {model_path}. "
                f"Check `model.name` in your experiment overrides."
            )
        with open(model_path) as f:
            data = yaml.safe_load(f)
        return data or {}

    def apply_overrides(
        self, data: dict, overrides: dict, *, strict: bool = True, allow_new_leaves: bool = True
    ) -> dict:
        """
        Apply dotted-path key-value overrides to a nested dictionary in-place.

        Args:
            data: Target dictionary to modify in-place.
            overrides: Dotted-key overrides (e.g. {"training.epochs": 50}).
            strict: If True, enforces that intermediate parent keys must already exist as dicts.
            allow_new_leaves: If True, allows setting new leaf keys that are not yet in the dict.

        Returns:
            The modified dictionary.
        """
        if not overrides:
            return data

        for dotted_key, value in overrides.items():
            parts = dotted_key.split(".")
            current = data
            traversed: list[str] = []

            # 1. Traverse intermediate parent dictionaries
            for part in parts[:-1]:
                traversed.append(part)

                if not isinstance(current, dict):
                    raise TypeError(
                        f"Override path '{dotted_key}' is invalid: "
                        f"parent '{'.'.join(traversed[:-1]) or '<root>'}' is not a dict "
                        f"(got {type(current).__name__})."
                    )

                if part not in current:
                    if strict:
                        raise KeyError(
                            f"Unknown override path '{dotted_key}': missing key "
                            f"'{'.'.join(traversed)}'."
                        )
                    current[part] = {}

                next_val = current[part]
                if strict and not isinstance(next_val, dict):
                    raise TypeError(
                        f"Override path '{dotted_key}' is invalid: "
                        f"'{'.'.join(traversed)}' is not a dict (got {type(next_val).__name__})."
                    )

                current = next_val

            # 2. Assign leaf value
            leaf = parts[-1]
            if not isinstance(current, dict):
                raise TypeError(
                    f"Override path '{dotted_key}' is invalid: "
                    f"parent '{'.'.join(parts[:-1]) or '<root>'}' is not a dict "
                    f"(got {type(current).__name__})."
                )

            if leaf not in current:
                if not allow_new_leaves:
                    raise KeyError(
                        f"Unknown override path '{dotted_key}': leaf key '{leaf}' does not exist at "
                        f"'{'.'.join(parts[:-1]) or '<root>'}'."
                    )
                parent = ".".join(parts[:-1]) or "<root>"
                print(
                    f"Notice: Adding new config key via override: '{dotted_key}' "
                    f"(parent: '{parent}', value: {value})"
                )

            current[leaf] = value

        return data

    def write_effective_configs(
        self,
        experiment_name: str,
        run_dict: dict,
        model_dict: dict,
        model_name: str,
    ) -> tuple[Path, Path]:
        """
        Write effective YAML files for an experiment under `configs/effective/<experiment_name>/`.

        Returns:
            Tuple of (effective_run_path, effective_model_path)
        """
        exp_dir = self.effective_root / _sanitize_experiment_name(experiment_name)
        exp_dir.mkdir(parents=True, exist_ok=True)

        run_path = exp_dir / "conf_run_effective.yaml"
        model_path = exp_dir / f"conf_model_{model_name}_effective.yaml"

        with open(run_path, "w") as f:
            yaml.dump(run_dict, f, default_flow_style=False)

        with open(model_path, "w") as f:
            yaml.dump(model_dict, f, default_flow_style=False)

        return run_path, model_path

    def generate_effective_paths(
        self,
        experiment_name: str,
        config_overrides: dict,
        model_overrides: dict,
        *,
        template: str | None = None,
    ) -> tuple[Path, Path]:
        """
        Generate effective run and model YAML files for a single experiment.

        Steps:
        1. Load baseline run config dictionary by template.
        2. Apply run config overrides (`updates.config`).
        3. Load baseline model config dictionary based on updated `model.name`.
        4. Apply model architecture overrides (`updates.model`).
        5. Save effective YAML files under `configs/effective/<experiment_name>/`.

        Returns:
            Tuple of (effective_run_path, effective_model_path).
        """
        run_dict = self.load_run_dict(template=template)
        self.apply_overrides(run_dict, config_overrides, strict=True)

        model_name = (run_dict.get("model") or {}).get("name", "HetGAT")
        baseline_model_path = self.resolve_model_config_path(run_dict)
        model_dict = self.load_model_dict(baseline_model_path)
        self.apply_overrides(model_dict, model_overrides, strict=True)

        return self.write_effective_configs(experiment_name, run_dict, model_dict, model_name)


def run_batch_experiments(
    experiment_configs: list[dict], config_manager: ConfigManager | None = None
) -> dict:
    """
    Run multiple experiments sequentially using generated effective YAML files.

    Args:
        experiment_configs: List of experiment configuration dictionaries from a suite file.
        config_manager: Optional ConfigManager instance.

    Returns:
        Dictionary mapping experiment names to execution status and error messages.
    """
    if config_manager is None:
        config_manager = ConfigManager()

    results: dict[str, dict] = {}

    for exp in experiment_configs:
        exp_name = exp["name"]
        template = exp.get("template")
        updates = exp.get("updates") or {}

        if not isinstance(updates, dict):
            raise TypeError(
                f"Experiment '{exp_name}' has invalid 'updates' type: "
                f"expected dict, got {type(updates).__name__}."
            )

        # Validate update scopes
        allowed_update_scopes = {"config", "model"}
        unexpected_scopes = set(updates.keys()) - allowed_update_scopes
        if unexpected_scopes:
            unexpected = ", ".join(sorted(unexpected_scopes))
            raise ValueError(
                f"Experiment '{exp_name}' has invalid update scope(s): {unexpected}. "
                f"Expected only: {sorted(allowed_update_scopes)}. "
                f"Use 'updates.config.<path>' for run config overrides and "
                f"'updates.model.<path>' for model config overrides."
            )

        config_updates = updates.get("config") or {}
        model_updates = updates.get("model") or {}

        if not isinstance(config_updates, dict):
            raise TypeError(
                f"Experiment '{exp_name}' has invalid 'updates.config' type: "
                f"expected dict, got {type(config_updates).__name__}."
            )

        if not isinstance(model_updates, dict):
            raise TypeError(
                f"Experiment '{exp_name}' has invalid 'updates.model' type: "
                f"expected dict, got {type(model_updates).__name__}."
            )

        description = exp.get("description", "")

        print(f"\n{'=' * 70}")
        print(f"Experiment: {exp_name}")
        if description:
            print(f"Description: {description}")
        print(f"{'=' * 70}")

        try:
            # Generate effective YAML configuration manifests for this experiment
            run_effective_path, model_effective_path = config_manager.generate_effective_paths(
                experiment_name=exp_name,
                config_overrides=config_updates,
                model_overrides=model_updates,
                template=template,
            )

            print(f"Effective run config:   {run_effective_path}")
            print(f"Effective model config: {model_effective_path}")

            # Load typed configuration and execute training run
            config, origin = load_config_with_origin(
                run_effective_path, model_config_path=model_effective_path
            )
            run_training(
                config,
                check_run=False,
                origin=origin,
            )

            results[exp_name] = {"status": "SUCCESS", "error": None}
            print(f"✓ {exp_name} completed successfully")

        except Exception as e:
            results[exp_name] = {"status": "FAILED", "error": str(e)}
            print(f"✗ {exp_name} failed: {e}")
            raise  # Stop batch execution on first error to allow immediate debugging

    return results


def load_experiment_suite(suite_path: Path) -> list[dict]:
    """
    Load an experiment suite YAML file and return the list of experiment configurations.
    """
    with open(suite_path) as f:
        suite = yaml.safe_load(f)

    return suite.get("experiments", [])


@click.command("train-suite")
@click.option(
    "-s",
    "--suite",
    type=click.Path(exists=True),
    required=True,
    help="Path to experiment suite YAML file (e.g. configs/experiment_suite_A.yaml).",
)
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True),
    default=None,
    help="Path to baseline config file to modify (defaults to configs/conf_run.yaml).",
)
def train_suite(suite: str, config: str) -> None:
    """
    Run a batch experiment suite from a YAML definition file.

    Generates effective YAML configurations under `configs/effective/<experiment_name>/`
    and executes training runs sequentially for each experiment in the suite.
    """
    suite_path = Path(suite)
    config_path = Path(config) if config else None

    print(f"Loading experiment suite from {suite_path}...")
    experiments = load_experiment_suite(suite_path)

    if not experiments:
        print("No experiments found in suite file.")
        return

    print(f"Found {len(experiments)} experiment(s) in suite")

    config_manager = ConfigManager(config_path)

    try:
        results = run_batch_experiments(experiments, config_manager)

        # Print final summary table of all experiments in the suite
        print(f"\n{'=' * 70}")
        print("BATCH SUITE SUMMARY:")
        print(f"{'=' * 70}")
        for exp_name, result in results.items():
            status = result["status"]
            if status == "SUCCESS":
                print(f"  ✓ {exp_name}: {status}")
            else:
                print(f"  ✗ {exp_name}: {status}")
                if result["error"]:
                    print(f"      Error: {result['error']}")

    except KeyboardInterrupt:
        print("\n\nBatch suite run interrupted by user.")
    except Exception as e:
        print(f"\n\nBatch suite run failed with error: {e}")
        raise


if __name__ == "__main__":
    train_suite()
