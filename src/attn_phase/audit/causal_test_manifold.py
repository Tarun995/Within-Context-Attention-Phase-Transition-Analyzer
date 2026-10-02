"""
audit/causal_test_manifold.py — on-manifold vs. off-manifold perturbation test.

BACKGROUND: in every causal test so far, wherever a significant difference
appeared, a random-vector control disrupted the model MORE than a real
donor representation. Hypothesis: real activations sit near a manifold the
model actually produces; matched-scale noise does not, so disruption should
scale with how in-distribution the injected vector is.

FIVE CONDITIONS, patched at the same prefix-boundary position:
  donor               real activation, DIFFERENT TruthfulQA question
                      (same "Q: ...\\nA:" template, same final token)
  unrelated_template  real activation, OFF-TOPIC question in the SAME
                      "Q: ...\\nA:" template (same final token, same format)
  unrelated           real activation from an off-topic plain SENTENCE
                      (different format AND different final token)
  shuffled_donor      the donor's own values in RANDOM dimension order
                      (identical norm and value distribution, no structure)
  noise               random vector matched in scale to the donor's std

WHY THE 4TH CONDITION EXISTS (a confound in the first version): the plain
"unrelated" sentences differ from the recipient in FORMAT and in the token
sitting at the patched position ("." vs the ":" of "A:"), not only in topic.
So donor < unrelated could reflect token/format mismatch rather than
manifold distance. unrelated_template isolates topic from format:
  donor ~= unrelated_template < unrelated < noise  -> the donor/unrelated gap
      was format/token mismatch, not topical distance.
  donor < unrelated_template < unrelated < noise   -> disruption tracks
      distance from the recipient's own context at several levels.
Either result is informative.

WHY shuffled_donor EXISTS: separates whether the donor's gentleness comes
from its SCALE (norm/std) or from dimension-specific STRUCTURE (e.g. the
few large-magnitude "rogue" dimensions documented in transformer hidden
states — see Timkey & van Schijndel 2021, and "massive activations" work
more recently). shuffled_donor keeps the exact same values and scale as
donor but destroys which value sits in which dimension.
  donor ~= shuffled_donor  -> scale/value distribution alone explains it,
      structure doesn't add anything further.
  donor << shuffled_donor  -> dimension-specific structure matters, not
      just the set of values present.

Disruption = L2 distance between choice-loglik vectors before/after patch
(continuous, better for ORDERING conditions than a binary flip) plus a
correctness-flip count for continuity with earlier results.
"""

import random

import numpy as np
import torch
from scipy.stats import wilcoxon

from attn_phase.audit.activations import capture_hidden_state_from_ids
from attn_phase.audit.causal_test import (
    predict_answer_patched, capture_prefix_hidden_state,
)

CONDITIONS = ["donor", "unrelated_template", "unrelated", "shuffled_donor",
              "noise"]

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

# Off-topic, but in the SAME "Q: ...\nA:" template as real prompts.
UNRELATED_TEMPLATE_QUESTIONS = [
    "What time does the bakery on the corner open in the morning?",
    "How long should I steep green tea before drinking it?",
    "Which paint is best for a wooden garden fence?",
    "Why is the highway so quiet on Sunday mornings?",
    "Where did the children leave their football yesterday?",
    "What is the best way to wrap a birthday present neatly?",
    "How often does the evening train usually run late?",
    "Can a cat sleep comfortably on a narrow windowsill?",
    "What colour should I choose for my kitchen curtains?",
    "Is it worth taking an umbrella on a cloudy afternoon?",
    "How do I keep bread fresh for more than two days?",
    "Which museum in town has the new painting exhibit?",
]


def capture_unrelated_vec(model, tokenizer, layer_idx, device, rng):
    """Hidden state at the last token of a random off-topic plain sentence."""
    sentence = rng.choice(UNRELATED_FILLER_SENTENCES)
    input_ids = tokenizer.encode(sentence, return_tensors="pt").to(device)
    return capture_hidden_state_from_ids(model, input_ids, layer_idx)


def shuffle_vec(vec, rng):
    """Random permutation of the vector's entries: EXACTLY the same values,
    norm and std as `vec`, but its dimension-specific structure (e.g. the
    few large-magnitude 'rogue' dimensions real GPT-2 activations have) is
    destroyed. Separates 'off-manifold because structure is wrong' from
    'off-manifold because scale/values are wrong'."""
    g = torch.Generator().manual_seed(rng.randrange(2 ** 31))
    perm = torch.randperm(vec.numel(), generator=g).to(vec.device)
    return vec[perm]


def disruption_magnitude(baseline_logliks, patched_logliks) -> float:
    a = np.array(baseline_logliks)
    b = np.array(patched_logliks)
    return float(np.linalg.norm(a - b))


def _safe_wilcoxon(a, b):
    if all(x == y for x, y in zip(a, b)):
        return 1.0
    return float(wilcoxon(a, b).pvalue)


_PAIRS = [
    ("donor", "unrelated_template"),
    ("unrelated_template", "unrelated"),
    ("donor", "unrelated"),
    ("unrelated", "noise"),
    ("donor", "noise"),
    ("donor", "shuffled_donor"),
    ("shuffled_donor", "noise"),
]


def run_manifold_test(model, tokenizer, question_pool, layer_idx, device,
                      n_questions=50, seed=0, mode="single", verbose=True):
    """Patch each sampled recipient's prefix position with all four vector
    types and compare disruption. `mode`: "single" or "range" (as in
    causal_test.py)."""
    from attn_phase.audit.activations import predict_answer

    rng = random.Random(seed)
    n_questions = min(n_questions, len(question_pool))
    recipients = rng.sample(question_pool, n_questions)
    # separate RNG for the shuffle so adding this condition does NOT change
    # which donors / filler texts the other conditions draw for a given seed
    shuffle_rng = random.Random(seed + 1)

    mags = {c: [] for c in CONDITIONS}
    flips = {c: 0 for c in CONDITIONS}

    for i, recipient in enumerate(recipients):
        base = predict_answer(model, tokenizer, recipient, device)

        donor_q = rng.choice([q for q in question_pool if q is not recipient])
        template_q = rng.choice(UNRELATED_TEMPLATE_QUESTIONS)
        vecs = {
            "donor": capture_prefix_hidden_state(
                model, tokenizer, donor_q.question, layer_idx, device),
            "unrelated_template": capture_prefix_hidden_state(
                model, tokenizer, template_q, layer_idx, device),
            "unrelated": capture_unrelated_vec(
                model, tokenizer, layer_idx, device, rng),
        }
        vecs["shuffled_donor"] = shuffle_vec(vecs["donor"], shuffle_rng)
        noise_rng = torch.Generator().manual_seed(shuffle_rng.randrange(2 ** 31))
        noise = torch.randn(vecs["donor"].shape, generator=noise_rng)
        vecs["noise"] = noise.to(vecs["donor"].device) * vecs["donor"].std()

        for cond, vec in vecs.items():
            res = predict_answer_patched(model, tokenizer, recipient,
                                         layer_idx, vec, device, mode=mode)
            mags[cond].append(disruption_magnitude(
                base["choice_logliks"], res["choice_logliks"]))
            flips[cond] += int(res["correct"] != base["correct"])

        if verbose and (i + 1) % 10 == 0:
            print(f"  ... {i + 1}/{n_questions} recipients tested")

    return {
        "n_questions": n_questions,
        "n_pairs": n_questions,
        "mean_disruption": {c: float(np.mean(mags[c])) for c in CONDITIONS},
        "flip_counts": flips,
        "wilcoxon_p": {f"{a}_vs_{b}": _safe_wilcoxon(mags[a], mags[b])
                       for a, b in _PAIRS},
        "raw_disruption": {c: list(v) for c, v in mags.items()},
    }