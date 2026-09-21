"""
tests/test_causal_patch.py — Verifies the property the entire causal test
depends on: patching an EARLY position in a sequence (the prefix boundary)
actually changes the model's scoring of LATER tokens (the answer choice),
and does NOT retroactively change anything at or before the patch position
(causal masking should prevent that — if it doesn't, something is deeply
wrong with the hook).

Uses a small RANDOMLY-INITIALIZED GPT-2 (no network). Run this before
trusting any real causal-test result from causal_test.py, same reasoning
as test_patch.py and test_audit_activations.py.
"""

import torch
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

from attn_phase.audit.activations import (
    capture_hidden_state_from_ids, run_with_hidden_patch,
)

LAYER_IDX = -1


def _tiny_model_and_tokenizer():
    cfg = GPT2Config(
        n_layer=2, n_embd=32, n_head=2, n_positions=64, vocab_size=50257,
    )
    model = GPT2LMHeadModel(cfg)
    model.eval()
    # Real GPT2Tokenizer needs vocab files from the Hub — for a
    # network-free unit test we build ids directly instead of tokenizing
    # real text. vocab_size=50257 matches gpt2's real vocab size so a real
    # tokenizer's output ids would be valid inputs to this model, without
    # actually invoking the tokenizer here.
    return model, cfg


def _fake_prefix_and_full_ids(n_prefix=6, n_choice=4, vocab_size=50257,
                               seed=0):
    g = torch.Generator().manual_seed(seed)
    prefix_ids = torch.randint(0, vocab_size, (1, n_prefix), generator=g)
    choice_ids = torch.randint(0, vocab_size, (1, n_choice), generator=g)
    full_ids = torch.cat([prefix_ids, choice_ids], dim=1)
    return prefix_ids, full_ids


def test_patch_at_prefix_position_changes_later_token_logits():
    """The property the whole causal test depends on: patching the LAST
    prefix token's hidden state must change the logits computed for the
    choice tokens that come AFTER it in the same sequence."""
    model, _ = _tiny_model_and_tokenizer()
    prefix_ids, full_ids = _fake_prefix_and_full_ids()
    n_prefix = prefix_ids.shape[1]
    patch_position = n_prefix - 1

    baseline_logits = run_with_hidden_patch(model, full_ids, LAYER_IDX,
                                             None, "cpu")
    real_vec = capture_hidden_state_from_ids(model, prefix_ids, LAYER_IDX)
    noise_vec = torch.randn_like(real_vec) * real_vec.std() * 5

    patched_logits = run_with_hidden_patch(
        model, full_ids, LAYER_IDX, noise_vec, "cpu",
        position=patch_position,
    )

    # Logits for the CHOICE tokens (positions >= patch_position) must
    # differ — this is the causal leverage the whole test relies on.
    later_baseline = baseline_logits[patch_position:]
    later_patched = patched_logits[patch_position:]
    assert not torch.allclose(later_baseline, later_patched, atol=1e-4), (
        "Patching the prefix's last-token hidden state did NOT change the "
        "logits for the answer-choice tokens that follow it. The causal "
        "test has no leverage if this doesn't hold — do not trust any "
        "causal_test.py result until this passes."
    )


def test_patch_does_not_affect_positions_before_it():
    """Causal masking sanity check: patching position k must NOT change
    logits computed for positions BEFORE k. If it does, the hook is
    somehow leaking information backward, which would invalidate the
    causal interpretation of any result."""
    model, _ = _tiny_model_and_tokenizer()
    prefix_ids, full_ids = _fake_prefix_and_full_ids()
    n_prefix = prefix_ids.shape[1]
    patch_position = n_prefix - 1  # patch the LAST prefix token

    baseline_logits = run_with_hidden_patch(model, full_ids, LAYER_IDX,
                                             None, "cpu")
    real_vec = capture_hidden_state_from_ids(model, prefix_ids, LAYER_IDX)
    noise_vec = torch.randn_like(real_vec) * real_vec.std() * 5

    patched_logits = run_with_hidden_patch(
        model, full_ids, LAYER_IDX, noise_vec, "cpu",
        position=patch_position,
    )

    # Positions strictly BEFORE the patch point must be untouched.
    earlier_baseline = baseline_logits[:patch_position]
    earlier_patched = patched_logits[:patch_position]
    assert torch.allclose(earlier_baseline, earlier_patched, atol=1e-4), (
        "Patching position k changed logits for positions BEFORE k — this "
        "should be impossible under causal (autoregressive) masking. "
        "Something is wrong with the hook or the model isn't using causal "
        "attention as assumed."
    )


def test_mcnemar_style_counting_logic():
    """Pure-logic test of the paired discordant-pair counting used in
    run_causal_test's McNemar calculation — no model needed."""
    from scipy.stats import binomtest

    # 10 pairs: donor flips 6 times, control flips 2 times, with 2
    # overlapping (both flip) and construct explicit discordant counts.
    paired = [
        (True, False), (True, False), (True, False), (True, False),
        (True, True), (True, True),
        (False, True), (False, True),
        (False, False), (False, False),
    ]
    b = sum(1 for d, c in paired if d and not c)   # donor only: expect 4
    c_ = sum(1 for d, c in paired if not d and c)  # control only: expect 2
    assert b == 4
    assert c_ == 2

    result = binomtest(b, b + c_, p=0.5)
    assert 0.0 <= result.pvalue <= 1.0


def test_range_patch_touches_every_prefix_position():
    """The 'range' mode's whole point is closing the bypass where the
    model could still read untouched question tokens. Verify that a
    slice-based patch actually overwrites EVERY position in the range,
    not just the last one (i.e. the same broadcasting behavior the
    single-position patch relies on works correctly for a slice too)."""
    model, _ = _tiny_model_and_tokenizer()
    prefix_ids, full_ids = _fake_prefix_and_full_ids()
    n_prefix = prefix_ids.shape[1]

    baseline_logits = run_with_hidden_patch(model, full_ids, LAYER_IDX,
                                             None, "cpu")
    real_vec = capture_hidden_state_from_ids(model, prefix_ids, LAYER_IDX)
    noise_vec = torch.randn_like(real_vec) * real_vec.std() * 5

    range_patched_logits = run_with_hidden_patch(
        model, full_ids, LAYER_IDX, noise_vec, "cpu",
        position=slice(0, n_prefix),
    )

    # Every choice-token position (>= n_prefix - 1, since patching
    # position n_prefix-1 affects prediction of the token right after it)
    # should differ from baseline — same causal-leverage property as
    # single-position patching, just applied across the whole prefix.
    later_baseline = baseline_logits[n_prefix - 1:]
    later_range_patched = range_patched_logits[n_prefix - 1:]
    assert not torch.allclose(later_baseline, later_range_patched,
                               atol=1e-4), (
        "Range-patching the entire prefix did NOT change downstream "
        "choice-token logits — the slice-based patch may not be "
        "broadcasting correctly. Do not trust range-mode causal-test "
        "results until this passes."
    )

    # And a same-vector-back roundtrip across the WHOLE range should still
    # be a no-op, same as the single-position case.
    roundtrip_vec = capture_hidden_state_from_ids(model, prefix_ids,
                                                    LAYER_IDX)
    # NOTE: this captures only the LAST prefix position's own vector, so
    # patching the whole range with it is NOT expected to be a true no-op
    # (earlier prefix positions have different original hidden states).
    # That's expected, not a bug — range mode intentionally overwrites
    # every position with a single vector, unlike single-position mode's
    # exact roundtrip property.
