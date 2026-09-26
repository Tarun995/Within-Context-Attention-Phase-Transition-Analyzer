"""
audit/causal_test_concentration.py — Causal test for Baseline 2
(attention-concentration score).

HOW THIS DIFFERS FROM baseline 1's CAUSAL TEST (causal_test.py), AND WHY:
Baseline 1's test pooled questions by CORRECTNESS (confidently-correct vs
confidently-incorrect) and asked "does patching change whether the
recipient's answer is correct?" That's the right test for a signal whose
claim IS "I predict correctness."

Baseline 2's claim is narrower and more specific: "how concentrated the
model's attention is, at the moment it's about to answer, predicts
correctness." So this test pools questions by CONCENTRATION SCORE itself
(high vs. low, top/bottom tercile) — not by correctness — and asks a more
direct question: if I transplant a high-concentration donor's residual-
stream representation into a low-concentration recipient (or vice versa),
does the recipient's OWN concentration score actually shift toward the
donor's? And does correctness shift too?

This needs TWO outcomes measured per pair, not one:
  1. concentration_shifted — did the recipient's attention-concentration
     score move toward the donor's, past some threshold, after patching?
  2. correctness_flipped — same as baseline 1's outcome, for comparability.

Reuses the EXACT SAME patch mechanism as baseline 1 (activations.
run_with_hidden_patch, patching the residual stream at the prefix
boundary) — the only thing that changes is how donor/recipient pools are
built and what gets measured afterward. A random-vector control and
McNemar's test are used for both outcomes, same rigor as baseline 1.

WHAT THIS DELIBERATELY DOES NOT DO: directly overwrite the model's raw
post-softmax attention weights mid-computation. That intervention is
significantly more fragile (version-dependent internal module structure)
and was explicitly left untested by this project's own Phase P1 (see
README Limitations: "not raw softmax weights"). This test instead asks
whether the REPRESENTATION that PRODUCES a given concentration pattern is
swappable — a legitimate, safer causal test of the same underlying claim,
but a distinct question from "does force-overriding attention weights
directly change behavior." Worth stating explicitly in any write-up
rather than letting the distinction blur.
"""

from dataclasses import dataclass

import torch

from attn_phase.audit.activations import run_with_hidden_patch
from attn_phase.audit.causal_test import capture_prefix_hidden_state
from attn_phase.audit.attention_score import compute_attention_concentration


@dataclass
class ConcentrationRecord:
    mcq: object
    score: float
    correct: bool


def build_concentration_pools(scores, labels, questions, tercile: float = 1 / 3):
    """
    Splits questions into a LOW-concentration pool and a HIGH-concentration
    pool by tercile of the raw concentration score — NOT by correctness,
    unlike baseline 1's pools. Each record still carries its own
    correctness label so correctness-flip can be measured alongside
    concentration-shift.
    """
    import numpy as np
    order = np.argsort(scores)
    n = len(scores)
    k = max(1, int(n * tercile))
    low_idx = order[:k]
    high_idx = order[-k:]

    low_pool = [ConcentrationRecord(mcq=questions[i], score=float(scores[i]),
                                     correct=bool(labels[i])) for i in low_idx]
    high_pool = [ConcentrationRecord(mcq=questions[i], score=float(scores[i]),
                                      correct=bool(labels[i])) for i in high_idx]
    return low_pool, high_pool


@dataclass
class ConcentrationCausalTestResult:
    n_pairs: int
    donor_concentration_shift_count: int
    control_concentration_shift_count: int
    donor_correctness_flip_count: int
    control_correctness_flip_count: int
    concentration_shift_p_value: float
    correctness_flip_p_value: float
    direction: str  # "low_to_high" or "high_to_low"


def run_concentration_causal_test(model, tokenizer, donor_pool: list,
                                   recipient_pool: list, layer_idx: int,
                                   device: str, n_pairs: int = 50,
                                   seed: int = 0,
                                   direction: str = "low_to_high",
                                   shift_threshold_frac: float = 0.5,
                                   read_layer_idx=None,
                                   verbose: bool = True
                                   ) -> ConcentrationCausalTestResult:
    """
    `direction`: "low_to_high" means donors come from the HIGH-concentration
    pool and recipients from the LOW pool (does injecting a "sharply
    focused" representation make a diffuse-attention recipient's own
    attention sharpen too?); "high_to_low" is the reverse.

    `shift_threshold_frac`: a concentration "shift" counts if the
    recipient's post-patch score moved at least this fraction of the way
    from its original score toward the donor's original score (0.5 = at
    least halfway). Prevents counting trivial noise-sized movements as a
    genuine shift.

    `read_layer_idx`: which layer's attention pattern to measure the
    concentration SHIFT in. MUST be a layer strictly AFTER `layer_idx`
    (the patched layer) — a transformer block's attention pattern is
    computed from its OWN INPUT, so patching a block's OUTPUT cannot
    possibly affect that same block's attention (there's nothing
    downstream of itself to influence). Defaults to -1 (the last layer),
    which works for any patched layer except the last one itself — if
    `layer_idx` is -1 (or otherwise equals `read_layer_idx`), this raises
    an error rather than silently measuring nothing, since that failure
    mode looks identical to a genuine null result but means the test had
    no leverage to begin with. Correctness-flip, by contrast, is always
    measurable regardless of which layer is patched, since logits are
    downstream of every block via the final layernorm + LM head.
    """
    if read_layer_idx is None:
        read_layer_idx = -1
    if read_layer_idx == layer_idx:
        raise ValueError(
            f"read_layer_idx ({read_layer_idx}) must differ from the "
            f"patched layer_idx ({layer_idx}) — a block's own attention "
            f"pattern is computed from its INPUT, so patching that "
            f"block's OUTPUT cannot affect its own attention pattern. "
            f"Pick a layer strictly after layer_idx to measure the "
            f"concentration shift in (e.g. read_layer_idx=-1 for the "
            f"last layer, unless layer_idx IS the last layer)."
        )

    import random
    from scipy.stats import binomtest

    rng = random.Random(seed)
    n_pairs = min(n_pairs, len(donor_pool), len(recipient_pool))
    donors = rng.sample(donor_pool, n_pairs)
    recipients = rng.sample(recipient_pool, n_pairs)

    donor_shift = 0
    control_shift = 0
    donor_flip = 0
    control_flip = 0
    shift_paired = []
    flip_paired = []

    for i, (donor, recipient) in enumerate(zip(donors, recipients)):
        donor_vec = capture_prefix_hidden_state(
            model, tokenizer, donor.mcq.question, layer_idx, device)
        noise_vec = torch.randn_like(donor_vec) * donor_vec.std()

        # Donor-patched pass: get logits (for correctness) AND attentions
        # (for concentration) in one forward call each.
        from attn_phase.audit.causal_test import _prefix_and_full_ids
        prefix_ids, _ = _prefix_and_full_ids(
            tokenizer, recipient.mcq.question,
            recipient.mcq.choices[0], device)
        patch_position = prefix_ids.shape[1] - 1

        _, donor_patched_attn = run_with_hidden_patch(
            model, prefix_ids, layer_idx, donor_vec, device,
            position=patch_position, return_attentions=True)
        _, control_patched_attn = run_with_hidden_patch(
            model, prefix_ids, layer_idx, noise_vec, device,
            position=patch_position, return_attentions=True)

        def _concentration_from_attn(attn_tuple):
            attn = attn_tuple[read_layer_idx]
            dist = attn[0, :, -1, :].clamp_min(1e-12)
            entropy_per_head = -(dist * dist.log()).sum(dim=-1)
            return -entropy_per_head.mean().item()

        donor_patched_score = _concentration_from_attn(donor_patched_attn)
        control_patched_score = _concentration_from_attn(control_patched_attn)

        target = donor.score
        start = recipient.score
        needed = abs(target - start) * shift_threshold_frac
        donor_shifted = abs(donor_patched_score - start) >= needed and \
            (donor_patched_score - start) * (target - start) > 0
        control_shifted = abs(control_patched_score - start) >= needed and \
            (control_patched_score - start) * (target - start) > 0

        # Correctness flip: reuse predict_answer_patched from causal_test.py
        from attn_phase.audit.causal_test import predict_answer_patched
        donor_result = predict_answer_patched(
            model, tokenizer, recipient.mcq, layer_idx, donor_vec, device)
        control_result = predict_answer_patched(
            model, tokenizer, recipient.mcq, layer_idx, noise_vec, device)
        donor_flipped = donor_result["correct"] != recipient.correct
        control_flipped = control_result["correct"] != recipient.correct

        donor_shift += int(donor_shifted)
        control_shift += int(control_shifted)
        donor_flip += int(donor_flipped)
        control_flip += int(control_flipped)
        shift_paired.append((donor_shifted, control_shifted))
        flip_paired.append((donor_flipped, control_flipped))

        if verbose and (i + 1) % 10 == 0:
            print(f"  ... {i + 1}/{n_pairs} pairs tested "
                  f"(concentration shifts — donor: {donor_shift}, "
                  f"control: {control_shift}; "
                  f"correctness flips — donor: {donor_flip}, "
                  f"control: {control_flip})")

    def _mcnemar_p(paired):
        b = sum(1 for d, c in paired if d and not c)
        c_ = sum(1 for d, c in paired if not d and c)
        if b + c_ == 0:
            return 1.0
        return binomtest(b, b + c_, p=0.5).pvalue

    return ConcentrationCausalTestResult(
        n_pairs=n_pairs,
        donor_concentration_shift_count=donor_shift,
        control_concentration_shift_count=control_shift,
        donor_correctness_flip_count=donor_flip,
        control_correctness_flip_count=control_flip,
        concentration_shift_p_value=_mcnemar_p(shift_paired),
        correctness_flip_p_value=_mcnemar_p(flip_paired),
        direction=direction,
    )
