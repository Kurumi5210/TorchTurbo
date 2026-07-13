"""Small dependency-light helpers shared by engine modules."""

_PEFT_PREFIX = "base_model.model."
_BASE_LAYER_SEGMENT = ".base_layer."
_ADAPTER_KEYWORDS = (
    ".lora_A.",
    ".lora_B.",
    ".lora_A2.",
    ".lora_B2.",
    ".lora_embedding_A.",
    ".lora_embedding_B.",
)


def strip_peft_prefix(name: str) -> str:
    if name.startswith(_PEFT_PREFIX):
        name = name[len(_PEFT_PREFIX):]
    return name.replace(_BASE_LAYER_SEGMENT, ".")


def is_peft_adapter_param(name: str) -> bool:
    return any(keyword in name for keyword in _ADAPTER_KEYWORDS)
