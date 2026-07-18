# Within-Context Attention Phase Transition Analyzer

**[Live dashboard →](https://within-context-attention-phase-tran.vercel.app/)** — interactive
view of the headline finding, the full research notebook (including the bugs found along the
way), and a panel for exploring your own `results.json` from a local run.

A statistical testing tool for a specific mechanistic-interpretability question: **does a
frozen transformer's attention behave measurably differently, within a single forward pass,
when it solves a task versus when it fails one — and if so, is that difference causal?**

This isn't a visualization tool — it's a hypothesis-testing pipeline. It generates matched
synthetic tasks, runs them through a HuggingFace causal LM, extracts attention-derived
metrics per token position, runs a properly powered, multiple-comparisons-corrected
statistical test comparing solved vs. failed task instances, and — as of Phase P1 —
causally tests the resulting correlation via activation patching.

## Headline result (GPT-2 small, Phase C1)

Across 30 task instances (5 seeds × 6 task types, matched for prompt length where relevant),
one of three tested metrics separates solved from failed tasks after Bonferroni correction:

| Metric | n(solved) | n(failed) | mean(solved) | mean(failed) | p (corrected) | effect size r | Result |
|---|---|---|---|---|---|---|---|
| **post_plateau_var** | 17 | 12 | 0.0371 | 0.0194 | **0.042** | **-0.549** | Solved > failed |
| plateau_onset_fraction | 17 | 13 | 0.4062 | 0.4738 | 1.000 | 0.186 | No effect |
| entropy_rise_rate | 17 | 13 | 0.0096 | 0.0106 | 1.000 | 0.140 | No effect |

**Reading it correctly:** solved tasks show *higher* post-plateau attention-entropy variance
than failed tasks — the model doesn't settle into a quieter, more stable attention pattern
when it succeeds. It keeps attention more dynamic and oscillatory, a plausible signature of
sustained pattern-matching rather than a static "locked-in" state. The other two tested
metrics show no significant separation — reported honestly, not dropped.

![Main Finding](plots/hero_finding.png)

## Is it causal? (Phase P1)

C1 shows a correlation. Phase P1 tested whether it's causal, via activation patching:
capturing the attention-layer output from a solved task's forward pass and splicing it into
a failed task's forward pass (and vice versa), then checking whether the model's answer
shifts.

Two pre-registered pilots — patching only the final prompt position, then patching the
entire post-plateau span (up to 149 of 166 positions in some pairs) — both found **0/10
pairs showed any shift**, in either direction. The patch mechanism itself was independently
verified to actually intervene in the forward pass before either result was trusted.

**Honest reading:** at GPT-2 small's last layer, task correctness is robust to this
component being heavily altered. That's real evidence `post_plateau_var` is a correlate of
solving rather than a cause of it, at least at this layer and via this component — not
evidence the model's attention dynamics are causally *irrelevant* everywhere. Earlier
layers, other components, and full-generation effects remain untested. Full method and
scope are in [`docs/FINDINGS.md`](docs/FINDINGS.md), Phase P1.

See [`docs/FINDINGS.md`](docs/FINDINGS.md) for the full narrative: the original (confounded)
finding, the two methodology bugs found and fixed, a third bug in the statistics code
that initially reported the C1 result's direction backwards, and the Phase P1 causal test
above in full. The same story is laid out interactively in the **Notebook** tab of the
[live dashboard](https://within-context-attention-phase-tran.vercel.app/).

## Why the C1 result can be trusted

Bonferroni correction was applied across all 3 tested metrics — including an earlier
24-comparison layer sweep, where a false positive at layer 10 did not survive correction
and was correctly discarded. The reported effect size (rank-biserial r = -0.549) is
medium-to-large, not borderline, and the raw per-task values were hand-verified against the
printed statistics table before being trusted (see FINDINGS.md, "Bug #3"). The same
discipline — pre-registering what counts as the test before running it, rather than
searching configurations until one looks significant — governed the Phase P1 patching
pilots above.

## Install

```bash
git clone https://github.com/Tarun995/Within-Context-Attention-Phase-Transition-Analyzer
cd Within-Context-Attention-Phase-Transition-Analyzer
pip install -e .
```

## Quickstart

```bash
attn-phase run --config configs/phase_c1.yaml
```

Loads GPT-2, builds 30 tasks across 6 task types (5 seeded instances each), runs the forward
passes, computes attention-derived metrics, saves a curves plot and results JSON to
`results/`, and prints the statistical table above.

Point the CLI at a different model or config directly:

```bash
attn-phase run --model gpt2-medium --layers 0-12 --seeds 5 --tasks all
```

Any HuggingFace causal LM name is accepted — see `configs/phase_c1.yaml` for the full set of
configurable fields.

Drop the `results.json` this produces into the **Run** tab of the
[live dashboard](https://within-context-attention-phase-tran.vercel.app/) to explore it —
task-by-task tables, a solved-vs-failed scatter plot, and the corrected statistical tests,
rendered from your own run.

Run the Phase P1 causal patching pilots directly:

```bash
python -m attn_phase.run_p1_pilot --n-pairs 5          # final-token patch
python -m attn_phase.run_p1_pilot_range --n-pairs 5     # full post-plateau span patch
```

Results write incrementally to `results/patch_manifest.csv` and
`results/patch_manifest_range.csv` — safe to interrupt and resume.

## Repository structure

```
attention-phase-analyzer/
    pyproject.toml
    configs/phase_c1.yaml
    dashboard/                       # source for the live dashboard (Vite + React)
    src/attn_phase/
        tasks.py                # synthetic task generation, multi-seed wrapper
        metrics.py               # attention entropy, plateau detection, oscillation metrics
        stats.py                    # Mann-Whitney U + rank-biserial effect size + Bonferroni
        runner.py                    # experiment orchestration
        layer_sweep.py                 # multi-layer variant
        patch.py                          # Phase P1: activation-patching hooks + manifest utils
        run_p1_pilot.py                    # Phase P1 pilot: final-token patching
        run_p1_pilot_range.py               # Phase P1 pilot: full post-plateau-span patching
        cli.py                                  # single command-line entry point
    tests/                        # 60+ tests: tasks, metrics, answer-matching, stats, patching
    docs/FINDINGS.md                # full research narrative, incl. bugs found + P1 causal test
    results/                          # generated at runtime, not tracked in git
```

## Testing

```bash
python -m pytest tests/ -v
```

60+ tests cover task generation, metric correctness on synthetic curves with known
properties, answer-matching regression cases, statistical direction-labeling (added after
Bug #3), and the Phase P1 patch-hook mechanism itself (`test_patch.py` — verifies the hook
actually intervenes in the forward pass rather than silently no-op'ing, since a broken hook
would look identical to a genuine null causal result).

## Limitations

- Single model tested end-to-end so far (GPT-2 small, 117M params); the CLI supports any
  HuggingFace causal LM but larger-model results are not yet reported.
- CPU-only run shown; not benchmarked for GPU throughput.
- One base seed (42) with 5 derived instances per task type — broader seed coverage is
  planned (see Future Work in FINDINGS.md).
- The `mod_arith_m10_d1` task type appears in *both* the solved and failed groups across
  different seed instances — the reported separation is partly at the instance level, not
  purely between task types. See FINDINGS.md for detail.
- **Causal patching (Phase P1) tested only GPT-2 small's last layer, only the attention
  module's output (not raw softmax weights, individual heads, or MLP output), and scored
  outcomes via next-token prediction rather than full multi-token generation.** The null
  result found is scoped to that specific configuration — see FINDINGS.md, Phase P1, for
  what remains untested and why it wasn't pursued further within this phase.

## Related work

- Olsson et al. (2022) — "In-context Learning and Induction Heads"
- Vig (2019) — BertViz, a multiscale attention visualization tool
- Edelman et al. (2024) — "The Evolution of Statistical Induction Heads: In-Context Learning Markov Chains" (NeurIPS 2024)
- Todd et al. (2024) — "Function Vectors in Large Language Models" (ICLR 2024)

## License

MIT