"""
stats.py — Statistical testing for the Attention Phase Analyzer.

Provides the Mann-Whitney U test infrastructure needed for Phase C1:
comparing plateau metric distributions between solved and failed tasks
with proper effect sizes and multiple-comparisons correction.

WHY MANN-WHITNEY U (not t-test):
    Sample sizes in Phase C1 are small (n=5 per task type, ~15-20 per
    group after grouping solved/failed). The t-test assumes normality,
    which cannot be verified at this sample size. Mann-Whitney U is a
    non-parametric rank-based test that makes no distributional assumption
    and is the standard choice for small neuroscience/interpretability
    samples.

WHY EFFECT SIZE (not just p-value):
    A p-value only tells you whether a difference is unlikely under the
    null. With small n, even a real effect may not reach p<0.05 (low
    power), and with large n, a trivially small effect can be "significant."
    Rank-biserial correlation (r) is the natural effect size for
    Mann-Whitney U: r=0 means no effect, r=1 means perfect separation,
    r=0.5 is conventionally "medium." We report both.

WHY BONFERRONI CORRECTION:
    Phase C1 tests multiple metrics (onset_fraction, post_plateau_var,
    has_plateau, entropy_rise_rate) and potentially multiple layers.
    Running k independent tests at alpha=0.05 gives a ~1-(0.95^k) chance
    of at least one false positive by chance alone. Bonferroni divides
    alpha by k, controlling the family-wise error rate. This is what
    prevented us from trusting the layer-10 ppvar "separation" seen in
    the layer sweep (which was 1 of 24 comparisons — exactly the
    multiple-comparisons trap).
"""

from dataclasses import dataclass
import numpy as np
from scipy.stats import mannwhitneyu


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class TestResult:
    """
    Result of a single Mann-Whitney U test between two groups.

    metric          : name of the metric tested
    layer           : layer index tested (None if last-layer-only analysis)
    n_solved        : number of solved-task observations
    n_failed        : number of failed-task observations
    solved_mean     : mean of solved group
    failed_mean     : mean of failed group
    u_statistic     : Mann-Whitney U statistic
    p_value         : raw two-sided p-value
    p_corrected     : Bonferroni-corrected p-value (p_value * n_comparisons)
    effect_size_r   : rank-biserial correlation in [-1, 1]
                      positive = solved < failed, negative = solved > failed
    significant     : True if p_corrected < alpha
    alpha           : significance threshold used (after correction)
    direction       : "solved > failed" or "solved < failed" or "no effect"
    """
    metric: str
    layer: int | None
    n_solved: int
    n_failed: int
    solved_mean: float
    failed_mean: float
    u_statistic: float
    p_value: float
    p_corrected: float
    effect_size_r: float
    significant: bool
    alpha: float
    direction: str


# ---------------------------------------------------------------------------
# Core test function
# ---------------------------------------------------------------------------

def mann_whitney_test(solved_values: list[float],
                       failed_values: list[float],
                       metric: str,
                       layer: int | None = None,
                       n_comparisons: int = 1,
                       alpha: float = 0.05) -> TestResult:
    """
    Run a two-sided Mann-Whitney U test comparing solved vs failed task
    distributions on a single metric, with Bonferroni correction.

    solved_values  : metric values for all solved-task instances
    failed_values  : metric values for all failed-task instances
    metric         : name of the metric (for reporting)
    layer          : layer index if this is a layer-sweep test, else None
    n_comparisons  : total number of tests being run in this analysis
                     (used for Bonferroni correction: p_corrected = p * k)
    alpha          : pre-correction significance level (default 0.05)

    Returns a TestResult dataclass. Inspect .significant and .effect_size_r
    for the key numbers. Never trust .significant alone — check .effect_size_r
    too. A p_corrected < 0.05 with |r| < 0.3 is a weak finding.
    """
    solved = [v for v in solved_values if v is not None and not np.isnan(v)]
    failed = [v for v in failed_values if v is not None and not np.isnan(v)]

    if len(solved) < 2 or len(failed) < 2:
        # Not enough data for a meaningful test
        return TestResult(
            metric=metric, layer=layer,
            n_solved=len(solved), n_failed=len(failed),
            solved_mean=float(np.mean(solved)) if solved else float("nan"),
            failed_mean=float(np.mean(failed)) if failed else float("nan"),
            u_statistic=float("nan"), p_value=float("nan"),
            p_corrected=float("nan"), effect_size_r=float("nan"),
            significant=False, alpha=alpha, direction="insufficient data",
        )

    u_stat, p_value = mannwhitneyu(solved, failed, alternative="two-sided")

    # Rank-biserial correlation: r = 1 - 2U / (n1 * n2)
    # Ranges from -1 (failed always > solved) to +1 (solved always > failed)
    n1, n2 = len(solved), len(failed)
    effect_r = float(1 - (2 * u_stat) / (n1 * n2))

    p_corrected = min(1.0, float(p_value) * n_comparisons)
    significant = p_corrected < alpha

    if not significant or abs(effect_r) < 0.1:
        direction = "no effect"
    elif effect_r > 0:
        direction = "solved < failed"
    else:
        direction = "solved > failed"

    return TestResult(
        metric=metric, layer=layer,
        n_solved=n1, n_failed=n2,
        solved_mean=float(np.mean(solved)),
        failed_mean=float(np.mean(failed)),
        u_statistic=float(u_stat),
        p_value=float(p_value),
        p_corrected=p_corrected,
        effect_size_r=effect_r,
        significant=significant,
        alpha=alpha,
        direction=direction,
    )


# ---------------------------------------------------------------------------
# Batch test across metrics and layers
# ---------------------------------------------------------------------------

def run_full_test_battery(results: list[dict],
                           metrics: list[str],
                           layers: list[int] | None = None,
                           alpha: float = 0.05) -> list[TestResult]:
    """
    Run Mann-Whitney U tests for every (metric, layer) combination with
    automatic Bonferroni correction for the total number of comparisons.

    results : list of task result dicts, each must have:
                - "solved": bool
                - for last-layer tests: metric values at top level
                  (e.g. result["plateau_onset_fraction"])
                - for layer-sweep tests: result["layers"][layer_idx][metric]
    metrics : list of metric names to test
              e.g. ["plateau_onset_fraction", "post_plateau_var",
                    "entropy_rise_rate"]
    layers  : list of layer indices to test, or None for last-layer-only
              (in which case metric values are read from the top-level dict)
    alpha   : pre-correction significance threshold

    Returns a list of TestResult objects, one per (metric, layer) pair,
    sorted by p_corrected ascending (strongest results first).
    """
    if layers is None:
        layers = [None]

    n_comparisons = len(metrics) * len(layers)
    test_results = []

    for layer in layers:
        for metric in metrics:
            solved_vals = []
            failed_vals = []

            for r in results:
                if layer is None:
                    val = r.get(metric)
                else:
                    val = r.get("layers", {}).get(layer, {}).get(metric)

                if r["solved"]:
                    solved_vals.append(val)
                else:
                    failed_vals.append(val)

            tr = mann_whitney_test(
                solved_vals, failed_vals,
                metric=metric, layer=layer,
                n_comparisons=n_comparisons,
                alpha=alpha,
            )
            test_results.append(tr)

    return sorted(test_results, key=lambda x: (
        x.p_corrected if not np.isnan(x.p_corrected) else 999))


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_test_results(test_results: list[TestResult],
                        show_all: bool = True) -> None:
    """
    Print a formatted summary table of test results.

    show_all : if False, only print significant results. If True (default),
               print all results sorted by p_corrected so the strongest
               signals are visible even if none cross the threshold.
    """
    to_show = test_results if show_all else [r for r in test_results
                                              if r.significant]
    if not to_show:
        print("No results to display.")
        return

    header = (f"{'metric':<28}{'layer':>7}{'n_s':>5}{'n_f':>5}"
              f"{'mean_s':>9}{'mean_f':>9}{'p_raw':>9}"
              f"{'p_corr':>9}{'r':>7}{'sig':>6}{'direction'}")
    print("\n" + "=" * 110)
    print("STATISTICAL TEST RESULTS — Mann-Whitney U, Bonferroni corrected")
    print("=" * 110)
    print(header)
    print("-" * 110)

    for r in to_show:
        layer_str = str(r.layer) if r.layer is not None else "last"
        sig_str = "YES" if r.significant else "no"
        print(
            f"{r.metric:<28}{layer_str:>7}{r.n_solved:>5}{r.n_failed:>5}"
            f"{r.solved_mean:>9.4f}{r.failed_mean:>9.4f}"
            f"{r.p_value:>9.4f}{r.p_corrected:>9.4f}"
            f"{r.effect_size_r:>7.3f}{sig_str:>6}  {r.direction}"
        )

    print("=" * 110)
    n_sig = sum(1 for r in test_results if r.significant)
    n_total = len(test_results)
    print(f"\n{n_sig} of {n_total} tests significant after Bonferroni correction.")

    if n_sig == 0:
        print("\nVERDICT: No metric separates solved from failed tasks at the "
              "corrected significance level.")
        print("This is a well-powered negative result. Report it honestly.")
    else:
        print("\nVERDICT: At least one metric shows significant separation.")
        print("Check effect_size_r — only trust results with |r| >= 0.3.")
        sig_results = [r for r in test_results if r.significant]
        for r in sig_results:
            print(f"  -> {r.metric} (layer={r.layer}): r={r.effect_size_r:.3f}, "
                  f"p_corrected={r.p_corrected:.4f}, direction={r.direction}")


def summarize_results_to_dict(test_results: list[TestResult]) -> list[dict]:
    """Convert TestResult objects to plain dicts for JSON serialization."""
    return [
        {
            "metric": r.metric,
            "layer": r.layer,
            "n_solved": r.n_solved,
            "n_failed": r.n_failed,
            "solved_mean": r.solved_mean,
            "failed_mean": r.failed_mean,
            "u_statistic": r.u_statistic,
            "p_value": r.p_value,
            "p_corrected": r.p_corrected,
            "effect_size_r": r.effect_size_r,
            "significant": r.significant,
            "alpha": r.alpha,
            "direction": r.direction,
        }
        for r in test_results
    ]