"""
test_metrics.py

Unit tests for metric functions in attn_phase.metrics.

Tests use synthetic curves with KNOWN properties so we can assert the
functions recover the right answer. This is the kind of test that would
have caught the original changepoint detector bug faster — that detector
was validated on synthetic decaying curves, but real GPT-2 curves rise.
Having explicit shape-specific tests makes that mismatch visible immediately.
"""

import numpy as np
import pytest
from attn_phase.metrics import (
    plateau_onset,
    post_plateau_variance,
    has_plateau,
    entropy_rise_rate,
    smooth,
    oscillation_envelope,
)


# ---------------------------------------------------------------------------
# Helpers: synthetic curve builders
# ---------------------------------------------------------------------------

def make_rising_plateau_curve(n=200, rise_end=60, plateau_val=2.0,
                               noise=0.05, seed=0):
    """Sigmoid-like rise then flat plateau — typical real GPT-2 shape."""
    rng = np.random.default_rng(seed)
    curve = np.zeros(n)
    for i in range(n):
        if i < rise_end:
            curve[i] = plateau_val * (i / rise_end)
        else:
            curve[i] = plateau_val
    curve += rng.normal(0, noise, n)
    return np.clip(curve, 0, None)


def make_still_rising_curve(n=200, noise=0.02, seed=0):
    """Curve that never plateaus — like sorting task."""
    rng = np.random.default_rng(seed)
    curve = np.linspace(0, 3.0, n)
    curve += rng.normal(0, noise, n)
    return np.clip(curve, 0, None)


def make_flat_curve(n=200, val=2.0, noise=0.02, seed=0):
    """Already-flat curve — onset should be very early."""
    rng = np.random.default_rng(seed)
    return np.full(n, val) + rng.normal(0, noise, n)


def make_oscillatory_plateau(n=200, rise_end=60, plateau_val=2.0,
                              oscillation_amp=0.3, seed=0):
    """Plateau with heavy oscillation — like failed mod_arith tasks."""
    rng = np.random.default_rng(seed)
    base = make_rising_plateau_curve(n, rise_end, plateau_val, noise=0.0)
    osc = oscillation_amp * np.sin(np.linspace(0, 20 * np.pi, n))
    return base + osc + rng.normal(0, 0.02, n)


# ---------------------------------------------------------------------------
# plateau_onset
# ---------------------------------------------------------------------------

class TestPlateauOnset:

    def test_rising_plateau_onset_in_correct_region(self):
        curve = make_rising_plateau_curve(n=200, rise_end=60, plateau_val=2.0)
        onset_pos, onset_frac = plateau_onset(curve, threshold_frac=0.90)
        assert onset_pos is not None
        # Should be detected somewhere in the rise region, not at the boundary
        assert onset_pos > 10
        assert onset_pos < 120   # shouldn't be pushed all the way to the end
        assert 0.0 < onset_frac < 1.0

    def test_still_rising_curve_onset_is_late(self):
        curve = make_still_rising_curve(n=200)
        onset_pos, onset_frac = plateau_onset(curve, threshold_frac=0.90)
        # A still-rising curve won't reach 90% of max until near the end
        assert onset_pos is not None
        assert onset_frac > 0.7

    def test_flat_curve_onset_is_early(self):
        curve = make_flat_curve(n=200, val=2.0)
        onset_pos, onset_frac = plateau_onset(curve, threshold_frac=0.90,
                                               causal_mask_region=10)
        assert onset_pos is not None
        assert onset_frac < 0.3   # flat from the start, onset should be early

    def test_returns_none_for_zero_curve(self):
        curve = np.zeros(100)
        onset_pos, onset_frac = plateau_onset(curve)
        assert onset_pos is None
        assert onset_frac is None

    def test_onset_fraction_consistent_with_position(self):
        curve = make_rising_plateau_curve(n=200)
        onset_pos, onset_frac = plateau_onset(curve)
        if onset_pos is not None:
            assert abs(onset_frac - onset_pos / len(curve)) < 1e-9


# ---------------------------------------------------------------------------
# post_plateau_variance
# ---------------------------------------------------------------------------

class TestPostPlateauVariance:

    def test_flat_plateau_has_low_variance(self):
        curve = make_rising_plateau_curve(n=200, noise=0.01)
        onset_pos, _ = plateau_onset(curve)
        ppv = post_plateau_variance(curve, onset_pos)
        assert ppv is not None
        assert ppv < 0.01   # very flat after onset

    def test_oscillatory_plateau_has_high_variance(self):
        curve = make_oscillatory_plateau(n=200, oscillation_amp=0.3)
        onset_pos, _ = plateau_onset(curve)
        ppv = post_plateau_variance(curve, onset_pos)
        assert ppv is not None
        assert ppv > 0.01   # oscillation should show up as high variance

    def test_returns_none_when_onset_is_none(self):
        ppv = post_plateau_variance(np.ones(100), onset_pos=None)
        assert ppv is None

    def test_returns_none_when_onset_too_close_to_end(self):
        curve = np.ones(20)
        ppv = post_plateau_variance(curve, onset_pos=17)
        assert ppv is None   # fewer than 5 positions remaining


# ---------------------------------------------------------------------------
# has_plateau
# ---------------------------------------------------------------------------

class TestHasPlateau:

    def test_flat_plateau_returns_true(self):
        curve = make_rising_plateau_curve(n=200, rise_end=60, noise=0.01)
        onset_pos, _ = plateau_onset(curve)
        assert has_plateau(curve, onset_pos) is True

    def test_still_rising_returns_false(self):
        curve = make_still_rising_curve(n=200)
        onset_pos, _ = plateau_onset(curve)
        assert has_plateau(curve, onset_pos) is False

    def test_none_onset_returns_false(self):
        curve = make_rising_plateau_curve(n=200)
        assert has_plateau(curve, onset_pos=None) is False


# ---------------------------------------------------------------------------
# entropy_rise_rate
# ---------------------------------------------------------------------------

class TestEntropyRiseRate:

    def test_rising_curve_has_positive_rate(self):
        curve = make_rising_plateau_curve(n=200, rise_end=60)
        rate = entropy_rise_rate(curve, causal_mask_region=10, rise_frac=0.25)
        assert rate is not None
        assert rate > 0

    def test_flat_curve_has_near_zero_rate(self):
        curve = make_flat_curve(n=200, noise=0.001)
        rate = entropy_rise_rate(curve, causal_mask_region=10, rise_frac=0.25)
        assert rate is not None
        assert abs(rate) < 0.05

    def test_returns_none_for_very_short_curve(self):
        curve = np.array([0.0, 1.0, 2.0])   # too short after masking
        rate = entropy_rise_rate(curve, causal_mask_region=10)
        assert rate is None


# ---------------------------------------------------------------------------
# smooth and oscillation_envelope (basic sanity checks)
# ---------------------------------------------------------------------------

class TestSmooth:

    def test_output_length_matches_input(self):
        curve = np.random.randn(150)
        assert len(smooth(curve, window=9)) == len(curve)

    def test_flat_curve_unchanged_by_smoothing(self):
        curve = np.ones(100) * 2.5
        smoothed = smooth(curve, window=9)
        np.testing.assert_allclose(smoothed, curve, atol=1e-10)

    def test_window_1_is_identity(self):
        curve = np.random.randn(50)
        np.testing.assert_allclose(smooth(curve, window=1), curve)


class TestOscillationEnvelope:

    def test_output_length_matches_input(self):
        curve = np.random.randn(100)
        env = oscillation_envelope(curve, window=15)
        assert len(env) == len(curve)

    def test_flat_curve_has_near_zero_envelope(self):
        curve = np.ones(100)
        env = oscillation_envelope(curve, window=15)
        np.testing.assert_allclose(env, 0.0, atol=1e-10)

    def test_sinusoidal_curve_has_nonzero_envelope(self):
        curve = np.sin(np.linspace(0, 10 * np.pi, 200))
        env = oscillation_envelope(curve, window=15)
        assert env.mean() > 0.5