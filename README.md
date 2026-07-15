# Within-Context Attention Phase Transition Analyzer

A statistical testing tool for a specific mechanistic-interpretability question: **does a
frozen transformer's attention behave measurably differently, within a single forward pass,
when it solves a task versus when it fails one?**

This isn't a visualization tool — it's a hypothesis-testing pipeline. It generates matched
synthetic tasks, runs them through a HuggingFace causal LM, extracts attention-derived
metrics per token position, and runs a properly powered, multiple-comparisons-corrected
statistical test comparing solved vs. failed task instances.

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

See [`docs/FINDINGS.md`](docs/FINDINGS.md) for the full narrative: the original (confounded)
finding, the two methodology bugs found and fixed, and a third bug in the statistics code
that initially reported this exact result's direction backwards.

## Why this result can be trusted

Bonferroni correction was applied across all 3 tested metrics — including an earlier
24-comparison layer sweep, where a false positive at layer 10 did not survive correction
and was correctly discarded. The reported effect size (rank-biserial r = -0.549) is
medium-to-large, not borderline, and the raw per-task values were hand-verified against the
printed statistics table before being trusted (see FINDINGS.md, "Bug #3").

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

## Repository structure

```
attention-phase-analyzer/
    pyproject.toml
    configs/phase_c1.yaml
    src/attn_phase/
        tasks.py        # synthetic task generation, multi-seed wrapper
        metrics.py       # attention entropy, plateau detection, oscillation metrics
        stats.py           # Mann-Whitney U + rank-biserial effect size + Bonferroni
        runner.py           # experiment orchestration
        layer_sweep.py        # multi-layer variant
        cli.py                  # single command-line entry point
    tests/                        # 55+ tests: tasks, metrics, answer-matching, stats
    docs/FINDINGS.md                # full research narrative, incl. bugs found
    results/                          # generated at runtime, not tracked in git
```

## Testing

```bash
python -m pytest tests/ -v
```

55+ tests cover task generation, metric correctness on synthetic curves with known
properties, answer-matching regression cases, and statistical direction-labeling — the last
category added specifically after Bug #3 (see FINDINGS.md).

## Limitations

- Single model tested end-to-end so far (GPT-2 small, 117M params); the CLI supports any
  HuggingFace causal LM but larger-model results are not yet reported.
- CPU-only run shown; not benchmarked for GPU throughput.
- One base seed (42) with 5 derived instances per task type — broader seed coverage is
  planned (see Future Work in FINDINGS.md).
- The `mod_arith_m10_d1` task type appears in *both* the solved and failed groups across
  different seed instances — the reported separation is partly at the instance level, not
  purely between task types. See FINDINGS.md for detail.

## Related work

- Olsson et al. (2022) — "In-context Learning and Induction Heads"
- Vig (2019) — BertViz, a multiscale attention visualization tool
- Edelman et al. (2024) — "The Evolution of Statistical Induction Heads: In-Context Learning Markov Chains" (NeurIPS 2024)
- Todd et al. (2024) — "Function Vectors in Large Language Models" (ICLR 2024)

## License

MIT
