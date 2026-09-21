"""
tests/test_audit_activations.py — Sanity checks for audit/activations.py's
hook mechanism, same reasoning as tests/test_patch.py: if the hook silently
no-ops, every causal-test pair would report "no shift," which looks
identical to a genuine null result but means nothing.

These tests use a small RANDOMLY-INITIALIZED GPT-2 config (no download, no
network required) — they test the hook plumbing, not the model's behavior.
They can run standalone before you ever touch real GPT-2 weights or the
TruthfulQA dataset, and should be run FIRST when setting this up on a new
machine, exactly like test_patch.py is meant to be for Phase P1.
"""

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from attn_phase.audit.activations import (
    capture_hidden_state_from_ids,
    run_with_hidden_patch,
)

LAYER_IDX = -1


def _tiny_model():
    cfg = GPT2Config(
        n_layer=2, n_embd=32, n_head=2, n_positions=64, vocab_size=100,
        bos_token_id=0, eos_token_id=0,
    )
    model = GPT2LMHeadModel(cfg)
    model.eval()
    return model


def _random_ids(seq_len: int = 10, vocab_size: int = 100, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab_size, (1, seq_len), generator=g)


def test_capture_returns_correct_shape():
    model = _tiny_model()
    input_ids = _random_ids()
    vec = capture_hidden_state_from_ids(model, input_ids, LAYER_IDX)
    assert vec.shape == (32,), (
        f"Expected [hidden_dim]=32, got {tuple(vec.shape)}"
    )


def test_patch_with_different_vector_changes_logits():
    model = _tiny_model()
    input_ids = _random_ids()
    device = "cpu"

    baseline_logits = run_with_hidden_patch(model, input_ids, LAYER_IDX,
                                             None, device)

    real_vec = capture_hidden_state_from_ids(model, input_ids, LAYER_IDX)
    noise_vec = torch.randn_like(real_vec) * real_vec.std() * 5

    patched_logits = run_with_hidden_patch(model, input_ids, LAYER_IDX,
                                            noise_vec, device)

    assert not torch.allclose(baseline_logits, patched_logits, atol=1e-4), (
        "Patching with a large random vector produced IDENTICAL logits. "
        "The hook is not affecting the forward pass — do not trust any "
        "causal-audit result until this passes. Likely cause: your "
        "installed transformers version doesn't route block output the "
        "way this module assumes."
    )


def test_patch_with_same_vector_is_a_noop():
    model = _tiny_model()
    input_ids = _random_ids()
    device = "cpu"

    baseline_logits = run_with_hidden_patch(model, input_ids, LAYER_IDX,
                                             None, device)

    real_vec = capture_hidden_state_from_ids(model, input_ids, LAYER_IDX)

    roundtrip_logits = run_with_hidden_patch(model, input_ids, LAYER_IDX,
                                              real_vec, device)

    assert torch.allclose(baseline_logits, roundtrip_logits, atol=1e-3), (
        "Patching with the model's OWN output vector changed the logits. "
        "The patch mechanism isn't a faithful replacement (check "
        "dtype/shape handling in run_with_hidden_patch) — do not trust "
        "any causal-audit result until this passes."
    )


def test_capture_is_deterministic_for_same_input():
    """Same input_ids through the same (eval-mode) model should always
    capture the same vector — catches accidental dropout-left-on or
    non-determinism that would corrupt the probe dataset silently."""
    model = _tiny_model()
    input_ids = _random_ids()

    vec1 = capture_hidden_state_from_ids(model, input_ids, LAYER_IDX)
    vec2 = capture_hidden_state_from_ids(model, input_ids, LAYER_IDX)

    assert torch.allclose(vec1, vec2), (
        "Capturing the same input twice gave different hidden states — "
        "check the model is in eval() mode (dropout must be off) before "
        "building the probe dataset."
    )
