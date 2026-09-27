"""
tests/test_causal_test_manifold.py — Sanity checks for
causal_test_manifold.py's math and statistics, using synthetic data and a
minimal fake tokenizer where a real model is needed. No network required.
"""

import numpy as np
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from attn_phase.audit.causal_test_manifold import (
    disruption_magnitude, capture_unrelated_vec,
    UNRELATED_FILLER_SENTENCES,
)


def _tiny_model():
    cfg = GPT2Config(n_layer=2, n_embd=32, n_head=4, n_positions=64,
                      vocab_size=50257, attn_implementation="eager")
    model = GPT2LMHeadModel(cfg)
    model.eval()
    return model


def test_disruption_magnitude_zero_when_identical():
    a = [-1.2, -0.8, -2.1, -0.4]
    assert disruption_magnitude(a, a) == 0.0


def test_disruption_magnitude_matches_manual_l2():
    a = [0.0, 0.0]
    b = [3.0, 4.0]
    # 3-4-5 triangle — exact, easy to hand-verify
    assert abs(disruption_magnitude(a, b) - 5.0) < 1e-9


def test_disruption_magnitude_orders_correctly():
    """Sanity-checks the core assumption the whole test relies on: a
    smaller perturbation should register a smaller magnitude than a
    larger one, for the same baseline."""
    baseline = [-1.0, -1.0, -1.0]
    small_shift = [-1.1, -0.9, -1.0]
    large_shift = [-3.0, 2.0, -0.5]

    small_mag = disruption_magnitude(baseline, small_shift)
    large_mag = disruption_magnitude(baseline, large_shift)
    assert small_mag < large_mag


def test_capture_unrelated_vec_uses_filler_sentences():
    """Confirms the function actually pulls from the filler-sentence pool
    (via a fake tokenizer that records what text it was asked to encode)
    rather than silently doing something else."""
    import random

    model = _tiny_model()
    seen_texts = []

    class _RecordingTokenizer:
        def encode(self, text, return_tensors=None):
            seen_texts.append(text)
            g = torch.Generator().manual_seed(abs(hash(text)) % (2**31))
            return torch.randint(0, 50257, (1, 8), generator=g)

    rng = random.Random(0)
    vec = capture_unrelated_vec(model, _RecordingTokenizer(), -1, "cpu", rng)

    assert vec.shape == (32,)
    assert seen_texts[0] in UNRELATED_FILLER_SENTENCES, (
        "capture_unrelated_vec did not use a sentence from the filler "
        "pool — check the rng.choice() call."
    )


def test_wilcoxon_detects_a_real_ordering():
    """Pure-statistics check: if donor magnitudes are CONSISTENTLY
    smaller than noise magnitudes across many paired samples, the
    Wilcoxon test should return a small p-value. If they're
    indistinguishable, it should not."""
    from scipy.stats import wilcoxon

    rng = np.random.default_rng(0)
    donor_mags = rng.normal(loc=1.0, scale=0.3, size=30)
    noise_mags = donor_mags + rng.normal(loc=2.0, scale=0.3, size=30)  # consistently larger

    p_real_difference = wilcoxon(donor_mags, noise_mags).pvalue
    assert p_real_difference < 0.01, (
        "Wilcoxon test failed to detect an obvious, consistent ordering "
        "between two paired samples — check the test setup."
    )

    similar_mags = donor_mags + rng.normal(loc=0.0, scale=0.3, size=30)
    p_no_difference = wilcoxon(donor_mags, similar_mags).pvalue
    assert p_no_difference > 0.05, (
        "Wilcoxon test found a 'significant' difference between two "
        "samples with no real systematic difference — check for a bug "
        "in how paired samples are constructed."
    )
