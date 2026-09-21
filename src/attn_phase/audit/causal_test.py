"""
audit/causal_test.py — The actual causal audit: does patching the hidden
state genuinely change which answer the model picks, or does the linear
probe's AUC (validated at layer 3, ~0.83 on TruthfulQA mc1 — see
linear_probe.py) merely correlate with correctness without being
load-bearing?

WHERE WE PATCH, AND WHY IT'S DIFFERENT FROM THE PROBE'S CAPTURE POINT:
linear_probe.py captures the hidden state at the end of "question +
predicted choice" — useful for PREDICTING correctness after the fact, but
patching that point can't causally influence which choice got picked,
because by then the choice is already fully written into the sequence and
its score already computed. Causal masking means a token's hidden state
only ever influences LATER tokens in the same forward pass, never earlier
ones.

So this module patches at the PREFIX boundary instead — the hidden state
right after "Q: {question}\\nA:", before any choice tokens exist. That
position causally feeds into the score of every choice token that follows
it, in the same forward pass, which is exactly the leverage a causal test
needs: patch it, then check whether the choice the model ends up favoring
actually changes.

METHOD:
    1. Capture each question's prefix hidden state at the validated layer
       (see build_probe_dataset — layer 3 confirmed as the AUC peak).
    2. Split questions into "confidently correct" and "confidently
       incorrect" pools, by loglik margin (how far ahead the winning
       choice was, not just whether it won).
    3. For each donor/recipient pair: patch the recipient's forward pass
       (for every choice) with the DONOR's prefix hidden state, re-score
       all choices, and check whether the recipient's predicted answer's
       correctness flipped.
    4. CONTROL, not just a raw flip count: also patch the same recipient
       with a random vector matched in scale to the donor vector. If
       donor-patching flips correctness more often than random-vector
       patching does, that's evidence of a genuine causal effect — not
       just "any large perturbation at this position breaks the model,"
       which would be a much weaker and less interesting finding.
    5. McNemar's test on the paired (donor-flip, random-flip) outcomes —
       the correct test for paired binary before/after comparisons, not a
       plain two-proportion test (donor and random patches are applied to
       the SAME recipient, not independent samples).

BEFORE TRUSTING ANY RESULT FROM THIS FILE: run
tests/test_causal_patch.py first. It verifies, with a random tiny model
(no network), that patching the PREFIX position genuinely changes
later-token scores — the one property this entire causal test depends on.
If that property doesn't hold on your installed transformers version, every
result from this module is meaningless, not just noisy.
"""

from dataclasses import dataclass

import torch

from attn_phase.audit.activations import (
    capture_hidden_state_from_ids, run_with_hidden_patch,
)
from attn_phase.audit.data import build_choice_prompt


def _prefix_and_full_ids(tokenizer, question: str, choice: str, device: str):
    prefix = f"Q: {question}\nA:"
    full = build_choice_prompt(question, choice)
    prefix_ids = tokenizer.encode(prefix, return_tensors="pt").to(device)
    full_ids = tokenizer.encode(full, return_tensors="pt").to(device)
    return prefix_ids, full_ids


def capture_prefix_hidden_state(model, tokenizer, question: str,
                                 layer_idx: int, device: str) -> torch.Tensor:
    """Captures the hidden state right after 'Q: {question}\\nA:' — the
    decision point BEFORE any answer choice exists, and the point this
    module's patching targets."""
    prefix = f"Q: {question}\nA:"
    prefix_ids = tokenizer.encode(prefix, return_tensors="pt").to(device)
    return capture_hidden_state_from_ids(model, prefix_ids, layer_idx)


def score_choice_loglik_patched(model, tokenizer, question: str, choice: str,
                                 layer_idx: int, patch_vec, device: str,
                                 mode: str = "single") -> float:
    """
    Same average-per-token teacher-forced log-likelihood as
    activations.score_choice_loglik, but with the hidden state at
    `layer_idx` overwritten by `patch_vec` (or left unpatched if patch_vec
    is None) before the choice tokens' scores are computed.

    `mode`:
      - "single" (default): patches only the LAST prefix token ("A:").
        This is a minimal intervention, but the model can potentially
        bypass it by re-reading the untouched question tokens directly
        via attention in later layers — the same limitation P1 already
        ran into with its own single-position patching strategy.
      - "range": patches EVERY position in the prefix (every question
        token, not just "A:") with the SAME patch_vec, broadcast across
        all of them. This closes that bypass — the recipient's entire
        pre-answer representation at this layer is replaced with a copy
        of the donor's summary vector, not just its final token. This is
        a blunter, stronger intervention by design: if a null result
        holds even under range-patching, that's much stronger evidence
        the signal isn't causally load-bearing at all, not just that a
        single position wasn't enough leverage.
    """
    prefix_ids, full_ids = _prefix_and_full_ids(tokenizer, question, choice,
                                                 device)
    n_prefix = prefix_ids.shape[1]
    n_choice_tokens = full_ids.shape[1] - n_prefix
    if n_choice_tokens <= 0:
        raise ValueError(
            f"Choice {choice!r} tokenized to 0 new tokens after the "
            f"question prefix — check formatting."
        )

    if mode == "single":
        patch_position = n_prefix - 1  # last token of the prefix, e.g. "A:"
    elif mode == "range":
        patch_position = slice(0, n_prefix)  # every prefix token
    else:
        raise ValueError(f"Unknown mode {mode!r} — use 'single' or 'range'.")

    logits = run_with_hidden_patch(model, full_ids, layer_idx, patch_vec,
                                    device, position=patch_position)
    log_probs = torch.log_softmax(logits, dim=-1)

    total_logprob = 0.0
    for pos in range(n_prefix, full_ids.shape[1]):
        target_id = full_ids[0, pos]
        total_logprob += log_probs[pos - 1, target_id].item()

    return total_logprob / n_choice_tokens


def predict_answer_patched(model, tokenizer, mcq, layer_idx: int, patch_vec,
                            device: str, mode: str = "single") -> dict:
    """Scores every choice in `mcq` with the SAME patch_vec applied at each
    choice's prefix position(s) (see `mode` in score_choice_loglik_patched),
    picks the argmax, reports correctness. Same shape of result as
    activations.predict_answer, for direct before/after comparison."""
    logliks = [
        score_choice_loglik_patched(model, tokenizer, mcq.question, choice,
                                     layer_idx, patch_vec, device, mode=mode)
        for choice in mcq.choices
    ]
    predicted_idx = max(range(len(logliks)), key=lambda i: logliks[i])
    return {
        "predicted_idx": predicted_idx,
        "correct": predicted_idx == mcq.correct_idx,
        "choice_logliks": logliks,
    }


@dataclass
class ConfidenceRecord:
    mcq: object          # data.MCQuestion
    predicted_idx: int
    correct: bool
    margin: float         # winning loglik minus runner-up loglik


def build_confidence_pools(model, tokenizer, questions: list, device: str,
                            verbose: bool = True):
    """
    Runs the model's own (unpatched) prediction on every question and
    splits them into a "confidently correct" pool and a "confidently
    incorrect" pool, sorted by margin (descending confidence). Margin, not
    just correct/incorrect, matters: a question the model barely got right
    isn't a good causal-test donor — we want clear cases to make the test
    meaningful.
    """
    from attn_phase.audit.activations import predict_answer

    correct_pool, incorrect_pool = [], []
    for i, mcq in enumerate(questions):
        result = predict_answer(model, tokenizer, mcq, device)
        sorted_logliks = sorted(result["choice_logliks"], reverse=True)
        margin = sorted_logliks[0] - sorted_logliks[1] if len(
            sorted_logliks) > 1 else float("inf")
        record = ConfidenceRecord(mcq=mcq, predicted_idx=result[
            "predicted_idx"], correct=result["correct"], margin=margin)
        (correct_pool if result["correct"] else incorrect_pool).append(
            record)

        if verbose and (i + 1) % 100 == 0:
            print(f"  ... {i + 1}/{len(questions)} questions scored")

    correct_pool.sort(key=lambda r: r.margin, reverse=True)
    incorrect_pool.sort(key=lambda r: r.margin, reverse=True)
    return correct_pool, incorrect_pool


@dataclass
class CausalTestResult:
    n_pairs: int
    donor_flip_count: int     # recipient's correctness flipped after DONOR patch
    control_flip_count: int   # recipient's correctness flipped after RANDOM patch
    donor_flip_rate: float
    control_flip_rate: float
    mcnemar_stat: float
    p_value: float
    direction: str             # "incorrect_to_correct" or "correct_to_incorrect"
    mode: str = "single"       # "single" or "range"


def run_causal_test(model, tokenizer, donor_pool: list, recipient_pool: list,
                     layer_idx: int, device: str, n_pairs: int = 50,
                     seed: int = 0, direction: str = "incorrect_to_correct",
                     mode: str = "single", verbose: bool = True
                     ) -> CausalTestResult:
    """
    Runs the donor -> recipient causal test. `direction` names what a
    "successful" patch would do: "incorrect_to_correct" means donors come
    from the CORRECT pool and recipients from the INCORRECT pool (does
    injecting a "confidently correct" representation fix a wrong answer);
    "correct_to_incorrect" is the reverse (does injecting a "confidently
    incorrect" representation break a right answer). Run both directions
    separately — a real causal signal should show up in at least one, and
    P1's own methodology treats a null result in either direction as a
    first-class finding, not a failure.

    `mode`: "single" (patch only the last prefix token) or "range" (patch
    every prefix token) — see score_choice_loglik_patched's docstring. Run
    "single" first; if it comes back null, "range" is the natural
    follow-up before concluding the signal isn't causal at all, same
    escalation P1 used (single-position, then post-plateau-range).
    """
    import random
    rng = random.Random(seed)

    n_pairs = min(n_pairs, len(donor_pool), len(recipient_pool))
    donors = rng.sample(donor_pool, n_pairs)
    recipients = rng.sample(recipient_pool, n_pairs)

    donor_flips = 0
    control_flips = 0
    # paired outcomes for McNemar: (donor_flipped, control_flipped) per pair
    paired = []

    for i, (donor, recipient) in enumerate(zip(donors, recipients)):
        donor_vec = capture_prefix_hidden_state(
            model, tokenizer, donor.mcq.question, layer_idx, device)
        noise_vec = torch.randn_like(donor_vec) * donor_vec.std()

        donor_result = predict_answer_patched(
            model, tokenizer, recipient.mcq, layer_idx, donor_vec, device,
            mode=mode)
        control_result = predict_answer_patched(
            model, tokenizer, recipient.mcq, layer_idx, noise_vec, device,
            mode=mode)

        # "Flip" means the recipient's correctness changed FROM its
        # original (unpatched) state, which was `recipient.correct`.
        donor_flipped = donor_result["correct"] != recipient.correct
        control_flipped = control_result["correct"] != recipient.correct

        donor_flips += int(donor_flipped)
        control_flips += int(control_flipped)
        paired.append((donor_flipped, control_flipped))

        if verbose and (i + 1) % 10 == 0:
            print(f"  ... {i + 1}/{n_pairs} pairs tested "
                  f"(donor flips so far: {donor_flips}, "
                  f"control flips so far: {control_flips})")

    # McNemar's test on the paired outcomes (exact binomial on discordant
    # pairs — appropriate for small n, no continuity-correction assumptions
    # needed).
    from scipy.stats import binomtest
    b = sum(1 for d, c in paired if d and not c)   # donor flipped, control didn't
    c_ = sum(1 for d, c in paired if not d and c)  # control flipped, donor didn't
    n_discordant = b + c_
    if n_discordant == 0:
        p_value = 1.0
        mcnemar_stat = 0.0
    else:
        result = binomtest(b, n_discordant, p=0.5)
        p_value = result.pvalue
        mcnemar_stat = b - c_

    return CausalTestResult(
        n_pairs=n_pairs,
        donor_flip_count=donor_flips,
        control_flip_count=control_flips,
        donor_flip_rate=donor_flips / n_pairs,
        control_flip_rate=control_flips / n_pairs,
        mcnemar_stat=mcnemar_stat,
        p_value=p_value,
        direction=direction,
        mode=mode,
    )
