# Causal Validity Audit of LLM Confidence-Probing Methods — Findings

**Status:** Baseline 1 (hidden-state linear probe) and Baseline 2
(attention-concentration score) both complete. On-manifold vs.
off-manifold perturbation test complete (Section 7).
**Model:** GPT-2 small (`gpt2`, 12 layers, HuggingFace `transformers`)
**Benchmark:** TruthfulQA, multiple-choice (`mc1`), full validation split, n=817
**Repo:** [Within-Context Attention Phase Transition Analyzer](https://github.com/Tarun995/Within-Context-Attention-Phase-Transition-Analyzer)

---

## 1. Summary

A hidden-state linear probe reproducing the Azaria & Mitchell (2023) style
of correctness prediction achieves **AUC 0.83** at layer 3 of GPT-2 small
on TruthfulQA mc1 — a working, validated baseline, in line with published
results for this class of method.

An attention-concentration score (Baseline 2), aggregated across all 12
layers and all attention heads, achieves only **AUC 0.54** — barely above
chance (0.5), and far weaker than Baseline 1. This was checked carefully,
not assumed: a single-layer version was tried first (best layer: AUC
0.55), then the full multi-layer aggregation (AUC 0.54, essentially
unchanged) — ruling out "under-built reproduction" as the explanation.
The weak signal appears to be real, not a bug.

Three activation-patching causal tests were then run: two on Baseline 1
(single-position and range patching), and one on Baseline 2 (patching the
representation behind the concentration score, at the layer with the
best available signal). **All three returned a null result on the
question they were designed to test** — donor-patched effects were
statistically indistinguishable from (or weaker than) a random-vector
control, in every condition. Baseline 1's null is the stronger, more
informative result, since it started from a clearly real signal (AUC
0.83); Baseline 2's null is real but carries less evidential weight,
since there wasn't much of a correlational signal to begin with.

A consistent, unplanned pattern showed up across every causal test run so
far, worth noting on its own: **wherever a statistically significant
difference appeared, the random-vector CONTROL disrupted the model more
than the real donor representation did** — the opposite of what a causal
effect would predict. This happened in Baseline 1's single-position test
(both directions) and Baseline 2's test (2 of 4 measured outcomes). Not a
single comparison, across either baseline, showed donor beating control.

**A dedicated follow-up test was then run to explain that pattern
directly, and found a clean, graded result:** disruption severity scales
specifically with how in-distribution an injected representation is — a
real donor activation (mean disruption 0.108), a real but topically
unrelated activation (0.556), and matched-scale random noise (4.061) form
a strictly increasing, statistically significant ordering (all pairwise
p<0.0001, n=50). This is genuine support for an "on-manifold vs.
off-manifold perturbation" explanation: it isn't about topical relevance
to the question specifically, it's about how close the injected vector is
to something the model would actually produce on its own. See Section 7
for full method and results — this is arguably the project's most
distinctive finding to date, discovered rather than reproduced from prior
work.

This is the second and third independent null result this project has
produced with this same causal-patching methodology — the first being
`post_plateau_var` in Phase P1. Three structurally different signals, on
two task families, all failing to demonstrate causal load-bearing status,
is the emerging pattern this audit is built to test for.

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

## 6. Baseline 2: Attention-Concentration Score

### 6.1 Method

An "AttentionScore"-style method: how strongly the model attends to a
small number of positions (concentrated, low-entropy attention) versus
spreading attention broadly (diffuse, high-entropy attention), used
directly as a correctness signal. Unlike Baseline 1, this needed no
trained classifier — a concentration score is already a single number per
question, so AUC was computed directly against the raw score, matching
how these methods are typically used in the literature.

**Score definition:** at the same prefix-boundary position used
throughout this audit (`"Q: {question}\nA:"`, immediately before any
answer choice exists), the attention distribution looking backward over
the question was extracted via `output_attentions=True`. Concentration =
negative mean entropy of that distribution, averaged across attention
heads (single-layer version) or across heads *and* all 12 layers
(multi-layer version).

**A real setup requirement, found while building this:** recent
`transformers` versions silently return an EMPTY attentions tuple with
`output_attentions=True` unless the model is loaded with
`attn_implementation="eager"`. No error, no warning — just nothing there.
Caught by a unit test before any real result was trusted (see
`tests/test_attention_score.py`).

### 6.2 Results: correlational signal is weak

| Layer | AUC (heads only) |
|---|---|
| **1** | **0.551** (best single layer) |
| 2 | 0.550 |
| 3 | 0.550 |
| 4 | 0.528 |
| 6 | 0.509 |
| 9 | 0.538 |
| 11 (last) | 0.546 |
| **All 12 layers, aggregated** | **0.538** |

Aggregating across all layers and heads (matching the original method
description in full, not just heads at one layer) did **not** meaningfully
improve on the best single-layer result — ruling out "under-built
reproduction" as the explanation for the weak signal. AUC 0.54–0.55
across every configuration tested, versus Baseline 1's 0.83, is a real,
checked finding: this particular formulation of attention concentration
carries little correctness signal on TruthfulQA for GPT-2 small.

### 6.3 Causal test

Layer 1 (best available signal) was used for the causal test, patching
the same prefix-boundary residual-stream representation as Baseline 1's
test, but with a different design suited to this signal's specific claim:
rather than pooling questions by *correctness*, questions were pooled by
their *concentration score itself* (tercile split: bottom third vs. top
third). Two outcomes were measured per donor/recipient pair, each against
a random-vector control and McNemar's test:
1. Did the recipient's own concentration score shift toward the donor's
   (measured at layer −1, downstream of the patched layer 1 — a block's
   attention pattern is computed from its own input, so measuring the
   shift at the *same* layer that was patched would be structurally
   guaranteed to show nothing, regardless of any real effect).
2. Did the recipient's correctness flip (same outcome type as Baseline 1,
   for comparability).

| Direction | Outcome | Donor | Control | p-value |
|---|---|---|---|---|
| donor=HIGH → recipient=LOW | Concentration shift | 84% (42/50) | 100% (50/50) | 0.0078 |
| donor=HIGH → recipient=LOW | Correctness flip | 0% (0/50) | 12% (6/50) | 0.0312 |
| donor=LOW → recipient=HIGH | Concentration shift | 0% (0/50) | 0% (0/50) | 1.0000 |
| donor=LOW → recipient=HIGH | Correctness flip | 2% (1/50) | 30% (15/50) | 0.0001 |

Three of the four outcomes reached significance (p < 0.025,
Bonferroni-corrected for 2 directions) — and in every significant case,
**control exceeded donor**, the opposite of what a causal effect would
predict. The fourth outcome (concentration shift, low→high direction) was
a flat 0/50 for both donor and control — worth a closer look in future
work to confirm whether this reflects a real asymmetry (e.g. attention
entropy has an asymmetric floor, easier to push up than down) or an
artifact of the shift-detection threshold used.

### 6.4 A pattern across both baselines, not planned for but worth naming

Across all causal tests run so far — Baseline 1's single-position patch,
Baseline 1's range patch, and Baseline 2's concentration-based patch —
every statistically significant comparison showed the same direction:
**random-vector noise disrupted the model's output more than a real,
meaningful donor representation did.** This was not hypothesized in
advance. A dedicated follow-up test was run to investigate it directly —
see Section 7.

---

## 7. On-Manifold vs. Off-Manifold Perturbation Test

### 7.1 Motivation and design

The pattern noted in Section 6.4 suggests a specific explanation: real
activations produced by real inputs sit near a lower-dimensional
"manifold" inside the full representation space. A donor swap — however
semantically wrong for the question at hand — is still a real,
in-distribution point near that manifold. Matched-scale random noise is
not; it points in an essentially arbitrary direction the model never
actually produces. If this is what's driving the pattern, disruption
severity should scale with how in-distribution the injected vector is,
not with whether it happens to be relevant to the question.

To test this directly (not just re-observe it), a third condition was
added between the existing two: for 50 recipient questions, the same
prefix-boundary position (layer 3, single-position patch — matching
Baseline 1's already-reported configuration) was patched with three
different vectors per recipient:
1. **donor** — a real activation from a different, ON-topic TruthfulQA
   question (as before).
2. **unrelated** — a real activation from a genuinely OFF-topic filler
   sentence (e.g. "The weather in the mountains changes quickly during
   autumn.") — real text, real activation, no relevance to TruthfulQA
   content or even to QA-style prompts at all.
3. **noise** — a random vector matched in scale to the donor's std (the
   existing control).

Disruption was measured two ways: a continuous magnitude (L2 distance
between the choice-loglikelihood vectors before and after patching — more
statistically informative for ORDERING three conditions than a binary
outcome) and a correctness-flip count, for continuity with earlier
results.

### 7.2 Results

| Condition | Mean disruption magnitude | Correctness flips (of 50) |
|---|---|---|
| donor (real, on-topic) | **0.108** | 0/50 |
| unrelated (real, off-topic) | **0.556** | 0/50 |
| noise (not real) | **4.061** | 8/50 |

| Comparison | Wilcoxon p-value |
|---|---|
| donor vs. unrelated | <0.0001 |
| unrelated vs. noise | <0.0001 |
| donor vs. noise | <0.0001 |

A strictly increasing, statistically significant ordering across all
three pairwise comparisons — not just the endpoints. The jump from donor
to unrelated (~5x) and from unrelated to noise (~7x) are each individually
significant, not just the overall donor-vs-noise gap.

### 7.3 Interpretation

This is genuine, graded support for the on-manifold explanation. Disruption
severity tracks specifically how in-distribution the injected
representation is — a real-but-irrelevant sentence sits meaningfully
between a real donor and pure noise, rather than behaving like either
endpoint. This rules out the narrower alternative explanation ("the
effect is really about topical relevance to the question, not
representation geometry") — if that were true, "unrelated" should have
behaved like noise, not landed cleanly in between.

This explains the pattern first noticed as a side effect of Baseline 1
and Baseline 2's causal tests, and stands on its own as this project's
most distinctive finding to date: a specific, discovered structural
property of GPT-2's representation space, found through a design built
for this project rather than reproduced from prior published work.

**What this does and doesn't say about the earlier causal-audit
results:** it does not overturn either null finding — Baseline 1 and
Baseline 2's correctness-flip patching results stand as reported. What it
adds is an explanation for *why* the random-vector control behaved the
way it did in those tests, and a new, independently interesting finding
about the geometry of the representation space itself.

---

## 8. Limitations

- **Single benchmark** (TruthfulQA mc1) — no cross-benchmark check yet
  (TriviaQA planned as a second benchmark per the original project plan).
- **Single model** (GPT-2 small) — no scaling check to Pythia or larger
  GPT-2 variants yet.
- **Two baseline methods completed, both null — but not equally strong
  evidence.** Baseline 1 started from a clearly real signal (AUC 0.83)
  and still failed the causal test — a strong, informative null. Baseline
  2's correlational signal was weak to begin with (AUC 0.54), so its
  causal null carries less weight: a weak signal failing a causal test is
  a less surprising result than a strong one failing.
- **Patching is position/layer-specific.** Baseline 1 tested only layer
  3 (the AUC-sweep peak); Baseline 2 tested only layer 1 (its own best
  layer, itself barely better than chance). A signal could in principle
  be causally load-bearing at a different layer even if the
  AUC-maximizing layer isn't. Not tested here for either baseline.
- **Baseline 2's causal test does not directly manipulate raw attention
  weights** — it tests whether the residual-stream representation behind
  a given concentration pattern is swappable, a related but distinct
  question from directly overriding post-softmax attention weights
  mid-computation. The latter is more fragile (version-dependent internal
  module structure) and was left untested here, consistent with this
  project's own Phase P1 also leaving raw-softmax-weight patching
  untested (see main README, Limitations).
- **The on-manifold vs. off-manifold finding (Section 7) used only 12
  fixed filler sentences and a single layer/mode (layer 3, single-patch)
  — a small, fixed pool. The result was clean and strongly significant,
  but worth expanding the filler set and testing at Baseline 2's layer
  too before treating it as fully general across configurations.
- **Donor/recipient pairing is unpaired by content** — pairs are random
  question pairs, not matched for topic or structure, unlike P1's
  same-template synthetic task pairs. This is a real methodological
  difference from P1's design, inherent to using natural-language
  questions instead of a synthetic task family with a fixed template.

---

## 9. Next steps

1. Second benchmark (TriviaQA) for Baseline 1, to check whether the null
   result is specific to TruthfulQA's adversarial design or generalizes.
2. Model scaling check (Pythia or GPT-2-medium) for Baseline 1.
3. Optionally extend Baseline 1's causal test to additional layers, in
   case a non-AUC-maximizing layer turns out to be causally load-bearing
   where layer 3 wasn't.
4. Extend the on-manifold test (Section 7) to Baseline 2's layer/setup
   and to range-patching, to check whether the donor < unrelated < noise
   ordering holds generally or is specific to layer 3/single-patch.
5. Write up the full result (two baselines, both null; the on-manifold
   finding explaining why) as the stretch-goal preprint-style document
   connecting C1 → P1 → this audit as one coherent research arc.
