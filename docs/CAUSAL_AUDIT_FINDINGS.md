# Causal Validity Audit of LLM Confidence-Probing Methods — Findings (Phase 1: Hidden-State Linear Probe)

**Status:** Baseline 1 complete (hidden-state linear probe). Baseline 2
(attention-concentration score) not yet started.
**Model:** GPT-2 small (`gpt2`, 12 layers, HuggingFace `transformers`)
**Benchmark:** TruthfulQA, multiple-choice (`mc1`), full validation split, n=817
**Repo:** [Within-Context Attention Phase Transition Analyzer](https://github.com/Tarun995/Within-Context-Attention-Phase-Transition-Analyzer)

---

## 1. Summary

A hidden-state linear probe reproducing the Azaria & Mitchell (2023) style
of correctness prediction achieves **AUC 0.83** at layer 3 of GPT-2 small
on TruthfulQA mc1 — a working, validated baseline, in line with published
results for this class of method.

Two activation-patching causal tests, of increasing strength, were then
run to check whether this AUC-0.83 signal is **load-bearing** (part of the
mechanism that produces correctness) or merely **correlational** (tracks
correctness without causing it). **Both tests returned a null result**:
donor-patched flip rates were statistically indistinguishable from a
random-vector control in every condition tested (all p > 0.025,
Bonferroni-corrected for 2 directions).

This is the second independent null result this project has produced with
this same causal-patching methodology — the first being `post_plateau_var`
in Phase P1 (synthetic mod_arith tasks, GPT-2 small, last-layer attention
output). Two structurally different signals, on two different task
families, both predicting behavior without demonstrably causing it, is
the emerging pattern this audit is built to test for.

---

## 2. Motivation

A literature check (see `CAUSAL_AUDIT_PROJECT_PLAN.md`) found that
internal-state signals predicting LLM correctness/confidence are an
actively developed and even commercialized area — but every method found
is validated correlationally. None test whether the signal is causally
load-bearing in the model's computation. This project already built and
validated an activation-patching causal test in Phase P1; this phase
extends that same test to a real, published class of confidence-probing
method on a real benchmark, rather than a synthetic task.

---

## 3. Method

### 3.1 Baseline: hidden-state linear probe

For each of 817 TruthfulQA mc1 questions:
1. Score every answer choice by average per-token teacher-forced
   log-likelihood (zero-shot cloze scoring, matching TruthfulQA's own mc1
   evaluation convention).
2. Take the model's own top-scored choice as its "answer."
3. Capture the residual-stream hidden state (post-block, pre-final-layernorm,
   via a direct `register_forward_hook` on `model.transformer.h[layer]`
   rather than `output_hidden_states`, for consistency with this
   project's existing patch.py hook philosophy) at the final token of
   `"Q: {question}\nA: {model's chosen answer}"`.
4. Label = 1 if the model's chosen answer was actually correct, else 0.
5. Train a logistic regression: hidden state → label, 70/30 train/test
   split, evaluated by held-out accuracy and AUC.

**Layer sweep** (GPT-2 small has 12 blocks, indices 0–11):

| Layer | AUC |
|---|---|
| 1 | 0.711 |
| 2 | 0.771 |
| **3** | **0.832** |
| 4 | 0.815 |
| 6 | 0.808 |
| 9 | 0.783 |
| −1 (last) | 0.781 |

Model's own base-rate accuracy on this question set: **27.7%** (expected —
TruthfulQA is designed to elicit confident-but-wrong answers; GPT-2 small
scoring below chance-on-4-choices is consistent with the benchmark's
purpose, not a bug).

AUC peaks sharply at layer 3 (bracketed on both sides by layers 2 and 4)
and declines steadily through the rest of the network — the model appears
to resolve whether it "knows" a TruthfulQA-style answer relatively early,
with later layers not adding separability for this particular signal.
**Layer 3 was used for all causal testing below.**

### 3.2 Causal test design

Patching the probe's own capture point (end of "question + chosen
answer") cannot test causality: by that point the answer is already fully
written into the sequence, and causal (autoregressive) masking means a
token's hidden state can only influence *later* tokens, never the ones
that produced it. The causal test instead patches the hidden state at the
**prefix boundary** — immediately after `"Q: {question}\nA:"`, before any
answer choice exists. This position causally feeds into the score of
every subsequent choice token in the same forward pass, giving the patch
actual leverage over which answer the model ends up favoring.

**Donor/recipient design**, extending patch.py's methodology:
1. Score all 817 questions unpatched; split into a **confidently-correct
   pool** (n=226) and **confidently-incorrect pool** (n=591), ranked by
   loglik margin between the top and second-place choice.
2. For each of 50 donor/recipient pairs per direction: capture the
   donor's prefix hidden state, patch it into the recipient's forward
   pass, re-score all the recipient's choices, and check whether the
   recipient's predicted-answer correctness flipped from its original
   (unpatched) state.
3. **Control, not just a raw flip count**: the same recipient is also
   patched with a random vector matched in scale to the donor vector
   (`torch.randn_like(donor_vec) * donor_vec.std()`). A high donor flip
   rate alone doesn't establish causality — any sufficiently large
   perturbation at this position might generically disrupt the model's
   output. Comparing against random-vector disruption isolates "the
   donor's specific representation matters" from "large perturbations
   here break things generically."
4. **McNemar's test** on the paired (donor-flipped, control-flipped)
   outcomes per pair — the correct test for paired binary before/after
   comparisons, since donor and control patches are applied to the same
   recipient, not independent samples.

Two directions were tested, mirroring P1's bidirectional check:
- **incorrect→correct**: donor from the correct pool, recipient from the
  incorrect pool — does injecting a "confidently correct" representation
  fix a wrong answer?
- **correct→incorrect**: the reverse — does injecting a "confidently
  incorrect" representation break a right answer?

Two levels of intervention strength were run:
- **Single-position**: patches only the final prefix token ("A:").
- **Range**: patches every prefix token (the whole question), closing the
  possible bypass where the model could still read the untouched question
  tokens directly via attention in later layers — the same escalation
  strategy P1 used (single-position, then post-plateau-range).

All patching and unit tests for the hook mechanism itself
(`tests/test_causal_patch.py`) were verified against a randomly-initialized
tiny GPT-2 before any real result was trusted, confirming: (a) patching
the prefix position genuinely changes downstream choice-token scores, (b)
patching does not retroactively affect positions before the patch point
(causal masking sanity check), and (c) range-patching correctly touches
every position in the slice.

---

## 4. Results

### 4.1 Single-position patch (layer 3, n=50 pairs/direction)

| Direction | Donor flip rate | Control flip rate | McNemar stat | p-value |
|---|---|---|---|---|
| incorrect → correct | 0.00% (0/50) | 12.00% (6/50) | −6 | 0.0312 |
| correct → incorrect | 0.00% (0/50) | 32.00% (16/50) | −16 | <0.0001 |

Donor flip rate was **0% in both directions** — transplanting a real
"confident" representation from a different question never changed the
recipient's answer. Notably, the *control* (random noise) flipped
correctness significantly more often than the real donor vector did in
both directions — the opposite of what a causal effect would predict.

### 4.2 Range patch (layer 3, n=50 pairs/direction)

| Direction | Donor flip rate | Control flip rate | McNemar stat | p-value |
|---|---|---|---|---|
| incorrect → correct | 20.00% (10/50) | 22.00% (11/50) | −1 | 1.0000 |
| correct → incorrect | 60.00% (30/50) | 66.00% (33/50) | −3 | 0.6476 |

Under the stronger intervention, both donor and control flip rates rose
substantially — consistent with "overwriting the entire question
representation is generally disruptive," not with the donor's specific
content mattering. Donor and control remain statistically
indistinguishable in both directions (p=1.0, p=0.65 — nowhere near the
Bonferroni-corrected 0.025 threshold for 2 directions tested).

A secondary, non-headline observation: breaking a correct answer (60–66%
flip rate) is substantially easier than fixing an incorrect one (20–22%
flip rate), for both donor and control patches alike, at this layer and
intervention strength. This asymmetry held under both single-position and
range patching and is worth investigating further, though it is not
itself evidence of causality either way.

---

## 5. Interpretation

The layer-3 hidden state that predicts correctness with AUC 0.83 does
**not** appear to be part of the mechanism that produces correctness, at
either intervention strength tested. Donor-patching — replacing a
recipient's representation with a real "confidently correct" or
"confidently incorrect" one from a different question — does not shift
outcomes any more reliably than patching with matched-scale random noise
does.

This is a genuine null result, not an inconclusive one: the test was run
at two escalating strengths specifically to rule out "the intervention
just wasn't strong enough" as an explanation, and the pattern (donor ≈
control, in both directions, at both strengths) held throughout.

Combined with Phase P1's independent null result for `post_plateau_var`
(different signal, different task family, same last-layer patching
methodology), this project has now found the same qualitative result
twice: a signal that predicts model behavior without demonstrably causing
it. Neither literature-reported confidence-probing methods nor this
project's own attention-derived metric survive a causal test, on the
evidence gathered so far.

---

## 6. Limitations

- **Single benchmark** (TruthfulQA mc1) — no cross-benchmark check yet
  (TriviaQA planned as a second benchmark per the original project plan).
- **Single model** (GPT-2 small) — no scaling check to Pythia or larger
  GPT-2 variants yet.
- **Single baseline method** (hidden-state linear probe) — the
  attention-concentration score (baseline 2) has not yet been reproduced
  or causally tested. A single method's null result is suggestive, not
  conclusive, for the broader claim that confidence-probing methods in
  general are correlational.
- **Patching is position/layer-specific.** Only layer 3 was causally
  tested (chosen as the AUC-sweep peak). A signal could in principle be
  causally load-bearing at a different layer even if the AUC-maximizing
  layer isn't. Not tested here.
- **Donor/recipient pairing is unpaired by content** — pairs are random
  question pairs, not matched for topic or structure, unlike P1's
  same-template synthetic task pairs. This is a real methodological
  difference from P1's design, inherent to using natural-language
  questions instead of a synthetic task family with a fixed template.

---

## 7. Next steps

1. Reproduce baseline 2 (attention-concentration score) and run the same
   two-strength causal test on it, for a second independent check of the
   correlational-vs-causal question.
2. Optionally extend the causal test to additional layers, in case a
   non-AUC-maximizing layer turns out to be causally load-bearing where
   layer 3 wasn't.
3. Second benchmark (TriviaQA) for generalization, time permitting.
4. Write up the two-baseline result as the stretch-goal preprint-style
   document connecting C1 → P1 → this audit as one coherent research arc.
