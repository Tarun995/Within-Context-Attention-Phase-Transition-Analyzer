"""
audit/causal_test_manifold.py — Tests the "on-manifold vs. off-manifold
perturbation" hypothesis, prompted by a pattern that showed up across
EVERY causal test run so far in this project (Baseline 1 single-patch,
Baseline 2's concentration test): wherever a statistically significant
difference appeared, a random-vector CONTROL disrupted the model's output
MORE than a real DONOR representation did — the opposite of what a causal
effect would predict.

ONE PLAUSIBLE EXPLANATION: real activations produced by real inputs sit
near a lower-dimensional "manifold" inside the full representation space.
A donor swap — however semantically wrong — is still a real, in-
distribution point near that manifold. Matched-scale random noise is not;
it points in an essentially arbitrary direction the model never actually
produces on its own. If that's what's driving the earlier pattern,
disruption severity should scale with how "in-distribution" the injected
vector is, not just with its raw magnitude.

THREE CONDITIONS, NOT TWO, TO TEST THIS DIRECTLY:
  1. donor       — a real activation from a DIFFERENT, ON-topic
                    TruthfulQA question (same as causal_test.py's donor).
  2. unrelated    — a real activation from an OFF-topic filler sentence:
                    real text, real activation, no relevance to the
                    question or to QA-style prompts at all.
  3. noise        — a random vector matched in scale to the donor's std
                    (not real at all — the existing control).

If the on-manifold hypothesis holds: donor < unrelated < noise, in
disruption severity. If "unrelated" behaves like noise instead of like
donor, the effect isn't about manifold geometry generally — it's
something more specific to QA-prompt structure.

DISRUPTION IS MEASURED TWO WAYS: a continuous magnitude (L2 distance
between the choice-loglikelihood vectors before and after patching — more
statistically informative than a binary outcome for ORDERING three
conditions) and a binary correctness-flip count (for continuity with
earlier results in this project).

VALIDATE BEFORE TRUSTING: run tests/test_causal_test_manifold.py first.
"""

import random

import numpy as np
import torch
from scipy.stats import wilcoxon

from attn_phase.audit.activations import capture_hidden_state_from_ids
from attn_phase.audit.causal_test import (
    predict_answer_patched, capture_prefix_hidden_state,
)

# Deliberately off-topic, unrelated to trivia/factual-QA content — the
# "real but presumably off the QA-prompt manifold" condition.
UNRELATED_FILLER_SENTENCES = [
    "The weather in the mountains changes quickly during autumn.",
    "She poured the tea slowly into a chipped ceramic cup.",
    "Traffic on the highway was unusually light this morning.",
    "The old wooden bridge creaked under every footstep.",
    "He spent the afternoon repainting the garden fence.",
    "A gentle breeze moved through the tall grass near the river.",
    "The bakery down the street sells fresh bread every morning.",
    "Children played football in the empty parking lot.",
    "The train arrived exactly three minutes behind schedule.",
    "Rain tapped softly against the window all evening.",
    "The museum's new exhibit features paintings from the 1800s.",
    "A cat stretched lazily on the warm windowsill.",
]


def capture_unrelated_vec(model, tokenizer, layer_idx: int, device: str,
                           rng: random.Random) -> torch.Tensor:
    """Captures a hidden state from a randomly chosen OFF-topic filler
    sentence, at that sentence's own final token position — real
    activation, real text, but unrelated to any TruthfulQA question or
    even to QA-style prompt structure at all."""
    sentence = rng.choice(UNRELATED_FILLER_SENTENCES)
    input_ids = tokenizer.encode(sentence, return_tensors="pt").to(device)
    return capture_hidden_state_from_ids(model, input_ids, layer_idx)


def disruption_magnitude(baseline_logliks, patched_logliks) -> float:
    """L2 distance between the choice-loglikelihood vectors before and
    after patching. A continuous measure of how much a patch disrupted
    the model's scoring, independent of whether it happened to flip the
    argmax choice — flips are a coarse, noisy signal for ORDERING three
    conditions by severity; this isn't."""
    a = np.array(baseline_logliks)
    b = np.array(patched_logliks)
    return float(np.linalg.norm(a - b))


def run_manifold_test(model, tokenizer, question_pool: list, layer_idx: int,
                       device: str, n_questions: int = 50, seed: int = 0,
                       mode: str = "single", verbose: bool = True) -> dict:
    """
    For each of `n_questions` sampled recipients, patches the SAME
    prefix-boundary position with three different vectors (donor,
    unrelated, noise) and compares disruption severity across all three,
    both as a continuous magnitude and as a correctness-flip count.

    `mode`: "single" or "range", same meaning as in causal_test.py —
    defaults to "single" to match the layer/mode combination already
    reported in the findings doc, for direct comparability.
    """
    from attn_phase.audit.activations import predict_answer

    rng = random.Random(seed)
    n_questions = min(n_questions, len(question_pool))
    recipients = rng.sample(question_pool, n_questions)

    donor_mags, unrelated_mags, noise_mags = [], [], []
    donor_flips, unrelated_flips, noise_flips = 0, 0, 0

    for i, recipient in enumerate(recipients):
        baseline_result = predict_answer(model, tokenizer, recipient, device)
        baseline_logliks = baseline_result["choice_logliks"]
        baseline_correct = baseline_result["correct"]

        candidates = [q for q in question_pool if q is not recipient]
        donor_q = rng.choice(candidates)
        donor_vec = capture_prefix_hidden_state(
            model, tokenizer, donor_q.question, layer_idx, device)
        unrelated_vec = capture_unrelated_vec(
            model, tokenizer, layer_idx, device, rng)
        noise_vec = torch.randn_like(donor_vec) * donor_vec.std()

        donor_result = predict_answer_patched(
            model, tokenizer, recipient, layer_idx, donor_vec, device,
            mode=mode)
        unrelated_result = predict_answer_patched(
            model, tokenizer, recipient, layer_idx, unrelated_vec, device,
            mode=mode)
        noise_result = predict_answer_patched(
            model, tokenizer, recipient, layer_idx, noise_vec, device,
            mode=mode)

        donor_mags.append(disruption_magnitude(
            baseline_logliks, donor_result["choice_logliks"]))
        unrelated_mags.append(disruption_magnitude(
            baseline_logliks, unrelated_result["choice_logliks"]))
        noise_mags.append(disruption_magnitude(
            baseline_logliks, noise_result["choice_logliks"]))

        donor_flips += int(donor_result["correct"] != baseline_correct)
        unrelated_flips += int(unrelated_result["correct"] != baseline_correct)
        noise_flips += int(noise_result["correct"] != baseline_correct)

        if verbose and (i + 1) % 10 == 0:
            print(f"  ... {i + 1}/{n_questions} recipients tested")

    def _safe_wilcoxon(a, b):
        if all(x == y for x, y in zip(a, b)):
            return 1.0
        return float(wilcoxon(a, b).pvalue)

    return {
        "n_questions": n_questions,
        "mean_disruption": {
            "donor": float(np.mean(donor_mags)),
            "unrelated": float(np.mean(unrelated_mags)),
            "noise": float(np.mean(noise_mags)),
        },
        "flip_counts": {
            "donor": donor_flips,
            "unrelated": unrelated_flips,
            "noise": noise_flips,
        },
        "n_pairs": n_questions,
        "wilcoxon_p": {
            "donor_vs_unrelated": _safe_wilcoxon(donor_mags, unrelated_mags),
            "unrelated_vs_noise": _safe_wilcoxon(unrelated_mags, noise_mags),
            "donor_vs_noise": _safe_wilcoxon(donor_mags, noise_mags),
        },
    }
