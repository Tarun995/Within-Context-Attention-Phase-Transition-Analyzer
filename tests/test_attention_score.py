"""
tests/test_attention_score.py — Sanity checks for attention_score.py and
causal_test_concentration.py, using a random tiny GPT-2 (no network) where
possible. Tokenizer-dependent pieces (compute_attention_concentration
itself takes real text) are exercised via raw input_ids instead, mirroring
how test_causal_patch.py handles the same constraint.
"""

import numpy as np
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from attn_phase.audit.activations import run_with_hidden_patch, \
    capture_hidden_state_from_ids
from attn_phase.audit.causal_test_concentration import build_concentration_pools

LAYER_IDX = -1


def _tiny_model():
    cfg = GPT2Config(n_layer=2, n_embd=32, n_head=4, n_positions=64,
                      # pyrefly: ignore [unexpected-keyword]
                      vocab_size=50257, attn_implementation="eager")
    # attn_implementation="eager" is REQUIRED for output_attentions=True to
    # actually return non-empty attention weights on recent transformers
    # versions — sdpa/flash-attention backends skip materializing them
    # even when asked. Without this, out.attentions comes back as an EMPTY
    # tuple, and a naive test loop over it silently passes without
    # checking anything (vacuous truth) rather than failing loudly. Real
    # run scripts must load the real model the same way — see
    # run_attention_score_probe.py.
    model = GPT2LMHeadModel(cfg)
    model.eval()
    return model


def _random_ids(seq_len=10, vocab_size=50257, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab_size, (1, seq_len), generator=g)


def _concentration_from_attn(attn_tuple, layer_idx):
    attn = attn_tuple[layer_idx]
    dist = attn[0, :, -1, :].clamp_min(1e-12)
    entropy_per_head = -(dist * dist.log()).sum(dim=-1)
    return -entropy_per_head.mean().item()


def test_unpatched_attentions_match_plain_forward_pass():
    """return_attentions=True with patch_vec=None should give identical
    attentions to a plain model call — confirms the new code path doesn't
    accidentally alter behavior for the unpatched case."""
    model = _tiny_model()
    ids = _random_ids()

    with torch.no_grad():
        plain_out = model(ids, output_attentions=True)

    assert len(plain_out.attentions) > 0, (
        "Model returned an EMPTY attentions tuple even with "
        "output_attentions=True — likely missing attn_implementation="
        "'eager' at model construction. Every test below would pass "
        "vacuously (looping over nothing) if this isn't caught here."
    )

    _, hook_attn = run_with_hidden_patch(model, ids, LAYER_IDX, None, "cpu",
                                          return_attentions=True)

    for a, b in zip(plain_out.attentions, hook_attn):
        assert torch.allclose(a[0], b, atol=1e-5), (
            "Attentions from run_with_hidden_patch(patch_vec=None) don't "
            "match a plain forward pass — the return_attentions path has "
            "a bug."
        )


def test_patching_changes_downstream_attention_pattern():
    """The causal leverage this whole test depends on: patching an EARLY
    layer's hidden state should change the ATTENTION PATTERN computed at
    a LATER layer. Deliberately does NOT read the concentration score
    from the SAME layer that was patched — a block's attention pattern is
    computed from its own INPUT, so patching that block's OUTPUT can
    never affect its own attention (see run_concentration_causal_test's
    read_layer_idx docstring for why this distinction matters)."""
    model = _tiny_model()  # 2 layers: patch layer 0, read layer 1 (-1)
    prefix_ids = _random_ids(seq_len=6)
    full_ids = torch.cat([prefix_ids, _random_ids(seq_len=4, seed=1)], dim=1)
    patch_position = prefix_ids.shape[1] - 1
    patch_layer = 0
    read_layer = -1  # the last (2nd) layer — downstream of layer 0

    real_vec = capture_hidden_state_from_ids(model, prefix_ids, patch_layer)
    noise_vec = torch.randn_like(real_vec) * real_vec.std() * 5

    _, baseline_attn = run_with_hidden_patch(
        model, full_ids, patch_layer, None, "cpu", return_attentions=True)
    _, patched_attn = run_with_hidden_patch(
        model, full_ids, patch_layer, noise_vec, "cpu",
        position=patch_position, return_attentions=True)

    baseline_score = _concentration_from_attn(baseline_attn, read_layer)
    patched_score = _concentration_from_attn(patched_attn, read_layer)

    assert baseline_score != patched_score, (
        "Patching an early layer did not change the LATER layer's "
        "attention-concentration score at all — the concentration causal "
        "test has no leverage if this doesn't hold."
    )


def test_same_layer_patch_and_read_shows_no_effect_as_expected():
    """Documents the bug this design deliberately avoids: patching layer
    L's OUTPUT and then reading layer L's OWN attention pattern shows NO
    effect, because that attention was already computed from layer L's
    INPUT before the patch ever applied. This is expected, not a failure
    — it's exactly why run_concentration_causal_test requires
    read_layer_idx != layer_idx and raises an error otherwise."""
    model = _tiny_model()
    prefix_ids = _random_ids(seq_len=6)
    full_ids = torch.cat([prefix_ids, _random_ids(seq_len=4, seed=1)], dim=1)
    patch_position = prefix_ids.shape[1] - 1
    same_layer = -1

    real_vec = capture_hidden_state_from_ids(model, prefix_ids, same_layer)
    noise_vec = torch.randn_like(real_vec) * real_vec.std() * 5

    _, baseline_attn = run_with_hidden_patch(
        model, full_ids, same_layer, None, "cpu", return_attentions=True)
    _, patched_attn = run_with_hidden_patch(
        model, full_ids, same_layer, noise_vec, "cpu",
        position=patch_position, return_attentions=True)

    baseline_score = _concentration_from_attn(baseline_attn, same_layer)
    patched_score = _concentration_from_attn(patched_attn, same_layer)

    assert baseline_score == patched_score, (
        "Expected patching a layer's output to have NO effect on that "
        "same layer's own attention pattern — if this now differs, "
        "something about the hook's timing relative to attention "
        "computation has changed and the read_layer_idx safeguard's "
        "reasoning needs to be re-checked."
    )


def test_build_concentration_pools_splits_correctly():
    """Pure-logic test: low pool should get the smallest scores, high pool
    the largest, with no model or tokenizer involved."""
    scores = np.array([5.0, 1.0, 9.0, 3.0, 7.0, 2.0, 8.0, 4.0, 6.0, 0.0])
    labels = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    questions = [f"q{i}" for i in range(10)]

    low_pool, high_pool = build_concentration_pools(
        scores, labels, questions, tercile=1 / 3)

    low_scores = sorted(r.score for r in low_pool)
    high_scores = sorted(r.score for r in high_pool)

    assert max(low_scores) < min(high_scores), (
        "Low-concentration pool contains a score higher than something in "
        "the high-concentration pool — tercile split logic is broken."
    )
    assert len(low_pool) == len(high_pool) == 3  # 10 * 1/3 -> 3 per tercile


def test_multilayer_concentration_matches_manual_per_layer_average():
    """The multi-layer aggregation function must give the SAME answer as
    manually computing each layer's concentration separately and
    averaging them — it's supposed to be a cheaper way to get an
    identical result (one forward pass instead of N), not a different
    computation. Uses a minimal fake tokenizer (fixed token ids, no
    vocab/network needed) so the REAL function under test actually runs,
    rather than duplicating its math in the test."""
    from attn_phase.audit.attention_score import (
        compute_attention_concentration_multilayer,
    )

    model = _tiny_model()  # 2 layers
    fixed_ids = _random_ids(seq_len=8)

    class _FakeTokenizer:
        def encode(self, text, return_tensors=None):
            # Ignores the actual text — always returns the same fixed ids,
            # so this test only needs to check the AGGREGATION logic, not
            # real tokenization.
            return fixed_ids

    with torch.no_grad():
        out = model(fixed_ids, output_attentions=True)
    manual_scores = []
    for layer_idx in [0, 1]:
        attn = out.attentions[layer_idx]
        dist = attn[0, :, -1, :].clamp_min(1e-12)
        entropy_per_head = -(dist * dist.log()).sum(dim=-1)
        manual_scores.append(-entropy_per_head.mean().item())
    expected_avg = sum(manual_scores) / len(manual_scores)

    actual = compute_attention_concentration_multilayer(
        model, _FakeTokenizer(), "irrelevant text", [0, 1], "cpu")

    assert abs(actual - expected_avg) < 1e-5, (
        f"Multi-layer function returned {actual}, expected {expected_avg} "
        f"from manual per-layer averaging — aggregation logic mismatch."
    )
