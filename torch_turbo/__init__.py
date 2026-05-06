# torch_turbo/__init__.py
"""
torch_turbo: Turbo-charged model implementations for efficient training and inference.

This package provides optimized model implementations with support for:
- Sequence Parallelism (Ulysses SP)
- Expert Parallelism (EP)
- FSDP2 distributed training
- NPU-optimized operators (for Ascend NPUs)

Usage:
    import torch_turbo

    # This will automatically register custom models/configs with transformers
    # if they are not already present in your transformers version.
"""

# Version
__version__ = "0.1.0"

# Flag to prevent duplicate registration
_REGISTERED = False


def register_model(config_class, model_class):
    """
    Register a custom model configuration and model with transformers Auto classes.

    This allows AutoConfig.from_pretrained() and AutoModelForCausalLM.from_pretrained()
    to automatically recognize and load custom model types from config.json.

    Uses the transformers registration API:
        AutoConfig.register(model_type, config_class, exist_ok=True)
        AutoModelForCausalLM.register(config_class, model_class, exist_ok=True)

    Args:
        config_class: The configuration class (e.g., GlmMoeDsaConfig)
                      Must have a `model_type` class attribute.
        model_class: The model class (e.g., GlmMoeDsaForCausalLM)

    Example:
        from torch_turbo import register_model
        from my_model import MyConfig, MyModelForCausalLM

        register_model(MyConfig, MyModelForCausalLM)

        # Now AutoConfig and AutoModelForCausalLM can load your model
        from transformers import AutoConfig, AutoModelForCausalLM
        config = AutoConfig.from_pretrained("path/to/model")
        model = AutoModelForCausalLM.from_pretrained("path/to/model")
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    # Get model_type from config class
    model_type = getattr(config_class, "model_type", None)
    if model_type is None:
        raise ValueError(
            f"Config class {config_class.__name__} must have a 'model_type' attribute. "
            "Please add `model_type = 'your_model_type'` to your config class."
        )

    # Register config: AutoConfig.register(model_type, ConfigClass)
    # exist_ok=True allows overwriting if already registered (for dev/testing)
    AutoConfig.register(model_type, config_class, exist_ok=True)
    print(f"[torch_turbo] Registered config: {config_class.__name__} for model_type='{model_type}'")

    # Register model: AutoModelForCausalLM.register(ConfigClass, ModelClass)
    # exist_ok=True allows overwriting if already registered (for dev/testing)
    AutoModelForCausalLM.register(config_class, model_class, exist_ok=True)
    print(f"[torch_turbo] Registered model: {model_class.__name__} for config {config_class.__name__}")


def _register_all_custom_models():
    """
    Internal: Register all custom models defined in torch_turbo.

    Add new models here by calling register_model().
    """
    # =================================================================
    # Register GlmMoeDsa (MLA with DSA indexer for sparse attention)
    # =================================================================
    import transformers

    # lower version transformers adapter
    if not hasattr(transformers, "TokenizersBackend"):
        from transformers import PreTrainedTokenizerFast
        class TokenizersBackend(PreTrainedTokenizerFast):
            def __init__(self, **kwargs):
                extra = kwargs.get("extra_special_tokens")
                if isinstance(extra, list):
                    kwargs.pop("extra_special_tokens")
                super().__init__(**kwargs)

        # Inject TokenizersBackend as an alias for PreTrainedTokenizerFast
        setattr(transformers, "TokenizersBackend", TokenizersBackend)

    if not hasattr(transformers, "GlmMoeDsaForCausalLM"):
        from .models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig
        from .models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaForCausalLM
        register_model(GlmMoeDsaConfig, GlmMoeDsaForCausalLM)

    # =================================================================
    # Register Deepseek V4
    # =================================================================
    if not hasattr(transformers, "DeepseekV4ForCausalLM"):
        from .models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
        from .models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4ForCausalLM
        register_model(DeepseekV4Config, DeepseekV4ForCausalLM)

    # =================================================================
    # Add more models here:
    # =================================================================
    # from .models.xxx.configuration_xxx import XxxConfig
    # from .models.xxx.modeling_xxx import XxxForCausalLM
    # register_model(XxxConfig, XxxForCausalLM)


def _auto_register():
    """
    Internal: Automatically register models when torch_turbo is imported.

    This is called at the end of __init__.py to ensure registration happens
    as soon as the package is imported.

    Example usage:
        # In your training script:
        import torch_turbo  # triggers auto-registration

        # Now AutoConfig and AutoModelForCausalLM can load custom models
        config = AutoConfig.from_pretrained("path/to/model")
        model = AutoModelForCausalLM.from_pretrained("path/to/model")
    """
    global _REGISTERED
    if _REGISTERED:
        return

    try:
        _register_all_custom_models()
        _REGISTERED = True
    except Exception as e:
        # Don't fail the entire import if registration fails
        # Users can manually call register_model() if needed
        import warnings
        warnings.warn(
            f"torch_turbo failed to register custom models with transformers: {e}\n"
            "You may need to manually call register_model() before loading models."
        )


# Expose registration function for manual use
__all__ = [
    "__version__",
    "register_model",
]



# =============================================================================
# Auto-register on import
# =============================================================================
# This ensures registration happens as soon as 'import torch_turbo' is executed
_auto_register()