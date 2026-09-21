"""
tests/test_audit_probe.py — Sanity checks for linear_probe.py's train/eval
logic, using synthetic hidden-state vectors (no model, no network). Confirms
the probe can recover a signal that's actually there, and that base_rate
reporting works, before trusting it on real captured activations.
"""

import numpy as np

from attn_phase.audit.linear_probe import ProbeDataset, train_and_evaluate_probe


def _make_separable_dataset(n=200, dim=16, seed=0):
    """Two well-separated Gaussian blobs — a probe SHOULD score near-perfect
    accuracy here. If it doesn't, something in the eval pipeline is broken,
    not the (nonexistent, by construction) signal."""
    rng = np.random.default_rng(seed)
    n_pos = n // 2
    n_neg = n - n_pos
    X_pos = rng.normal(loc=3.0, scale=1.0, size=(n_pos, dim))
    X_neg = rng.normal(loc=-3.0, scale=1.0, size=(n_neg, dim))
    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * n_pos + [0] * n_neg)
    qids = list(range(n))
    return ProbeDataset(X=X, y=y, question_ids=qids)


def _make_random_dataset(n=200, dim=16, seed=0):
    """Label has NO relationship to X — a probe should score close to the
    base rate / chance here, not spuriously high."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, dim))
    y = rng.integers(0, 2, size=n)
    qids = list(range(n))
    return ProbeDataset(X=X, y=y, question_ids=qids)


def test_probe_recovers_separable_signal():
    dataset = _make_separable_dataset()
    result = train_and_evaluate_probe(dataset)
    assert result["accuracy"] > 0.95, (
        f"Probe scored {result['accuracy']:.2f} accuracy on a trivially "
        f"separable synthetic dataset — expected >0.95. Check the "
        f"train/test split or classifier setup before trusting real results."
    )
    assert result["auc"] is not None and result["auc"] > 0.95


def test_probe_does_not_hallucinate_signal_from_noise():
    dataset = _make_random_dataset()
    result = train_and_evaluate_probe(dataset)
    # With no real signal, expect accuracy near chance (0.5) +/- sampling
    # noise on a modest test set — not near-perfect.
    assert result["accuracy"] < 0.75, (
        f"Probe scored {result['accuracy']:.2f} accuracy on label-shuffled "
        f"noise with no real signal — suspiciously high, check for a "
        f"train/test leak (e.g. duplicate rows, or fitting on test data)."
    )


def test_base_rate_reported_correctly():
    dataset = _make_separable_dataset(n=100)
    result = train_and_evaluate_probe(dataset)
    assert abs(result["base_rate"] - 0.5) < 0.05, (
        "base_rate should reflect the dataset's actual positive-label "
        "fraction (~0.5 for this balanced synthetic set)."
    )
