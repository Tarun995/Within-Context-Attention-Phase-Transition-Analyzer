# Findings — Full Research Narrative

This document traces every claim made in the README back to a specific experiment, bug,
or fix. Nothing here has been smoothed over — including the parts that didn't work.

## Phase 0 — Initial finding (later found to be confounded)

The original experiment tested 4 synthetic tasks on GPT-2 small using a changepoint-based
methodology: attention entropy and an induction-head "oscillation envelope" were tracked
per token position, and a changepoint detector was used to find structural transitions.

| Task | Tokens | Solved | Entropy inflection | Induction envelope growth |
|---|---|---|---|---|
| copy_60 | 242 | No | 68 | 4.8% |
| copy_40 | 162 | No | 68 | Flat |
| mod_arith_50 | 504 | Yes | 137 | 29.9% |
| transduction_40 | 251 | No | 67 | Flat |

This was presented as evidence that successful task-solving is associated with a later,
more prolonged internal attention reorganization.

**The confound:** look at the token counts. `mod_arith_50` — the only solved task — is
also the only task at 504 tokens; every failed task is 162–251 tokens. The later
"inflection point" for the solved task is exactly what raw prompt length would predict on
its own, independent of whether the model solved anything. Solved-status and prompt length
were never separated in this design, so the finding could not distinguish "the model solved
it" from "the prompt was longer."

**A second, more basic problem, worth naming separately from the confound:** Phase 0's
conclusion was drawn from a single solved-task instance versus three failed ones. Even set
aside the length confound entirely — n=1 for the "solved" condition cannot support a general
claim about how successful algorithmic behavior relates to attention dynamics. This is why
Phase C1 moved to 5 independently-seeded instances per task type (17 solved, 12–13 failed
after grouping) and a proper Mann-Whitney U test — the original conclusion wasn't just
confounded, it was underpowered to begin with.

This finding was presented at the 3rd Doctoral Symposium 2026, NIIT University in April 2026 before the confound
below was identified; the poster is archived at `docs/historical_results/` for traceability
and should not be read as the project's current conclusion.

## Phase 0.5 — Deconfounding

Length-matched task variants were built specifically to break the confound: `copy_matched_504`
and `transduction_matched_504` — both padded/constructed to match the 504-token length of
the modular arithmetic task, so that token count is no longer a proxy for solved/failed
status. These variants are part of the current Phase C1 task set.

## Bug #1 — Answer-matching scorer

The function responsible for judging whether the model's generated output counts as
"solved" had a scoring bug — found and fixed during this phase. (See `test_answer_matching.py`
for the regression tests covering the specific cases this bug touched: bare integer answers,
single-token answers, and multi-token answers.)

## Bug #2 — Changepoint detector unreliable on real curves

The changepoint detector used in Phase 0 had been validated only on synthetic curves that
didn't resemble the actual shape of real attention-entropy curves in practice. Once tested
against real model output, its detected "changepoints" were not reliable. It was kept in
the codebase for reference (documented as unreliable) but abandoned as the primary metric.
Replacement: plateau-based metrics — `plateau_onset_fraction`, `post_plateau_var`,
`has_plateau` — which measure where and how stably a curve settles, rather than trying to
detect a discrete jump.

## The layer-10 false positive

An early full layer-sweep (12 layers × 2 metrics = 24 comparisons) found an apparently
significant separation at layer 10. Run through Bonferroni correction across all 24
comparisons, it did not survive — a textbook multiple-comparisons false positive. This is
the direct reason Phase C1's statistical testing (`stats.py`) applies Bonferroni correction
by default across every metric tested in a single run, rather than treating each metric's
raw p-value in isolation.

## Bug #3 — Direction-label sign error in stats.py

Phase C1's Mann-Whitney U test correctly found a significant separation on
`post_plateau_var` (p_corrected = 0.042, r = -0.549). But the automated `direction` label
printed **"solved < failed"** — while the reported group means showed `mean(solved) = 0.0371`
and `mean(failed) = 0.0194`, i.e. solved was clearly *higher*.

Pulling the raw per-task values from the results JSON and sorting them by group confirmed
the means were correct and the direction label was inverted. Root cause: rank-biserial
correlation is computed as `r = 1 - 2U / (n1 * n2)` with solved as the first group. This
formula produces a **negative** r when the first group's values are *larger* — the opposite
of the intuitive reading. The code's `direction` branch had the sign backwards
(`effect_r > 0 → "solved > failed"` when it should be `effect_r > 0 → "solved < failed"`).

This was verified independently by recomputing U and r directly from the raw per-task
values (not just re-reading the code), confirming: `U = 158.0`, `effect_r = -0.549`,
`mean(solved) = 0.0371 > mean(failed) = 0.0194` — negative r does correspond to
"solved > failed" for this data. The fix was a one-line swap of the two direction strings,
verified against the full 55-test suite (no regressions) and a fresh experiment re-run
producing identical statistics with the corrected label.

**This changes the interpretation, not just the label.** The corrected finding is: solved
tasks maintain *more* dynamic, oscillatory attention after the plateau point, not a
quieter/more stable one. A regression test (`test_stats.py`) was added specifically to
prevent this class of bug from reappearing silently.

## Final Phase C1 result (corrected)

- **Metric:** `post_plateau_var`
- **n(solved) = 17, n(failed) = 12** (some instances excluded for missing values, e.g.
  `sorting_seed42002_inst2`)
- **mean(solved) = 0.0371, mean(failed) = 0.0194**
- **U = 158.0, p_raw = 0.0140, p_corrected = 0.0420 (Bonferroni, k=3)**
- **effect_size_r = -0.549** → **direction: solved > failed**
- **plateau_onset_fraction and entropy_rise_rate: no significant effect** (reported
  honestly, not omitted)

**Caveat:** the `mod_arith_m10_d1` task type appears in both groups — seed instances 0 and
4 were solved, instances 1–3 were failed. The separation found is therefore partly at the
level of individual task instances (same task, different seed, different outcome), not
purely between distinct task types. This is a meaningful nuance, not a confound to hide —
same-task-different-outcome variance is itself informative.

## Phase P1 — Causal validation via activation patching (null result)

C1 established that `post_plateau_var` *correlates* with solving. Phase P1 asked the next
question: does it *cause* solving, or is it a side effect? Two pre-registered activation-
patching pilots were run to test this, using `mod_arith_m10_d1` (the task type with a
genuine solved/failed mix, per the caveat above), GPT-2 small's last layer (layer -1,
matching C1's default), and 5 patch pairs per direction (10 total per pilot).

**Patch target:** the attention module's output (`attn_output` — the value-weighted,
output-projected vector added to the residual stream), not raw post-softmax attention
weights. This target was chosen deliberately over patching the softmax weights directly:
overwriting those mid-computation requires depending on internal `transformers`-library
attention implementation details that vary across versions (eager vs. SDPA attention
paths), whereas `attn_output` is the module's stable public return value and is directly
downstream of the weights the project's metrics measure. Before trusting any patching
result, the hook mechanism itself was verified with a dedicated test
(`tests/test_patch.py`): confirmed that patching with a different vector changes the
model's output logits, and that patching with the model's own unmodified vector is a
no-op — ruling out a silently-inert hook, which would look identical to a genuine null
result but mean nothing.

**Pilot 1 — final token only:** patched only the attention output at the last prompt
position (the position next-token prediction is directly conditioned on).
Result: **0/10 pairs showed any shift** in next-token correctness, in either direction.

**Pilot 2 — full post-plateau span:** patched every position from each recipient
instance's own plateau-onset position (computed identically to how `post_plateau_var`
itself locates plateau onset) through the end of the sequence — between 24 and 149
positions per pair, in several cases nearly the entire sequence.
Result: **0/10 pairs showed any shift**, even with most of the sequence's post-onset
attention output replaced.

**Conclusion:** at GPT-2 small's last layer, task correctness is robust to having the
post-plateau attention-layer output partially or almost entirely replaced by a donor
instance's. This is a materially stronger null than either pilot alone — the same result
held whether one position or nearly the whole post-plateau span was patched. It is strong
evidence that `post_plateau_var`, while a real and statistically significant correlate of
solving (C1), is **not causally driven by this layer's attention output** — at least not
in a way this patching design can detect.

**Scope, stated explicitly rather than implied:** this rules out the last layer's
attention-output component specifically, tested via a next-token-prediction proxy. It does
not test earlier layers, other components (MLP outputs, individual attention heads), or
effects that only appear across full multi-token generation rather than single-token
prediction. Per the project's own rule against iterating configurations until one shows
significance (see "What not to do," repo README/roadmap), these are noted as untested and
deliberately not pursued further within this phase — a third patching configuration was
not run to go looking for a positive result.

## Future work

- ~~Activation patching across the plateau boundary~~ — **done, Phase P1 above (null
  result at the last layer, attention-output component)**
- Activation patching at earlier layers, or on other components (MLP output, individual
  attention heads) — Phase P1 tested only the last layer's attention output
- Full multi-token generation scoring for patching, rather than the next-token proxy used
  in Phase P1
- Head-level ablation to identify which attention heads drive the `post_plateau_var` signal
- Larger model comparisons (GPT-2 medium/large, other model families)
- Broader seed coverage beyond base seed 42
- Layerwise propagation analysis of the plateau signal across all 12 layers, properly
  Bonferroni-corrected from the start (avoiding a repeat of the layer-10 false positive)

## Related work

- Olsson et al. (2022) — "In-context Learning and Induction Heads"
- Vig (2019) — BertViz, a multiscale attention visualization tool
- Edelman et al. (2024) — "The Evolution of Statistical Induction Heads: In-Context Learning Markov Chains" (NeurIPS 2024)
- Todd et al. (2024) — "Function Vectors in Large Language Models" (ICLR 2024)