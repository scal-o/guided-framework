from ml_static.config import Config, TrainingFreezeConfig


def generate_run_name(config: Config) -> str:

    if config.training.mode == "finetune":
        finetune_strategy = generate_finetune_strategy_string(config.training.freeze)
        return f"{config.raw_config['model']['name']} on {config.dataset.name} ({config.training.mode}) [{finetune_strategy}]"
    else:
        return f"{config.raw_config['model']['name']} on {config.dataset.name} ({config.training.mode})"


def generate_finetune_strategy_string(freeze_config: TrainingFreezeConfig) -> str:
    """
    Maps the freeze configuration to a concise strategy string.

    Args:
        config (dict): The configuration dictionary containing the 'training' block.

    Returns:
        str: A pipe-delimited string representing the unfrozen components.
    """

    # if freeze is not enabled, it's a full fine-tuning strategy
    if not freeze_config.enabled:
        return "Full"

    exclude = freeze_config.exclude
    keep = freeze_config.keep

    # if exclude is empty, all components are frozen: this is a zero-shot strategy
    if not exclude:
        return "Zero-Shot"

    unfrozen_components = []

    # Preprocessing
    if "node_initializer" not in exclude:
        unfrozen_components.append("Preproc")

    # Encoders
    if "encoders" in exclude:
        # Check specific layers kept frozen vs excluded
        # Assuming encoders.0 is V-Enc and encoders.1 is R-Enc based on the YAML structure
        v_enc_unfrozen = any("encoders.0" in k for k in keep)
        r_enc_unfrozen = any("encoders.1" in k for k in keep)

        # If 'encoders' is in exclude, but specific layers are kept frozen,
        # it implies partial unfreezing. If no specific layers are kept frozen,
        # it implies full unfreezing of that encoder block.

        if v_enc_unfrozen:
            unfrozen_components.append("Last-V-Enc")
        # Add logic here if you ever unfreeze ALL of V-Enc while keeping R-Enc frozen

        if r_enc_unfrozen:
            unfrozen_components.append("Last-R-Enc")

    # Predictor Head
    if "predictor" not in exclude:
        unfrozen_components.append("Head")

    return " | ".join(unfrozen_components)
