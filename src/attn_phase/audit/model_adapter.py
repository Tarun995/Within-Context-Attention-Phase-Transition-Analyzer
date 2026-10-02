"""
audit/model_adapter.py — lets the block capture/patch primitive in
activations.py work on more than GPT-2.

Every other module in this package (linear_probe, causal_test,
causal_test_concentration, causal_test_manifold) reaches transformer
blocks only through activations._get_block_module, so generalizing that
one lookup makes the whole pipeline architecture-agnostic.

Supported: GPT-2 family (model.transformer.h) and Pythia / GPT-NeoX
family (model.gpt_neox.layers). Anything else raises instead of guessing:
a wrong block reference would make every captured or patched vector
meaningless with no error to signal it.
"""


def _blocks(model):
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers
    raise ValueError(
        f"Don't know where transformer blocks live for "
        f"{type(model).__name__}. Supported: GPT-2 family "
        f"(model.transformer.h) and Pythia/GPT-NeoX family "
        f"(model.gpt_neox.layers). Add a case in model_adapter._blocks "
        f"before using this model with the causal-audit pipeline."
    )


def get_block_module(model, layer_idx: int):
    return _blocks(model)[layer_idx]


def get_num_layers(model) -> int:
    return len(_blocks(model))


def auto_layers(num_layers: int) -> list:
    """A spread of layers: first, 25%, 50%, 75%, last. For GPT-2 small
    (12 layers) this gives [0, 3, 6, 9, 11], which includes layer 3, the
    layer the earlier single-layer results were reported at."""
    n = num_layers
    return sorted({0, n // 4, n // 2, (3 * n) // 4, n - 1})