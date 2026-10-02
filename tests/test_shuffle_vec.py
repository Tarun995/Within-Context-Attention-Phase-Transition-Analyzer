"""
tests/test_shuffle_vec.py — checks shuffle_vec's three required properties:
same values (just reordered), same norm/scale, and determinism per seed.
Pure math, no model or network needed.
"""
import random

import torch

from attn_phase.audit.causal_test_manifold import shuffle_vec


def test_shuffle_preserves_values_and_norm_but_reorders():
    v = torch.randn(64)
    v[5] = 40.0   # mimic a "rogue"/outlier dimension
    v[17] = -25.0

    shuffled = shuffle_vec(v, random.Random(0))

    assert torch.equal(torch.sort(shuffled).values, torch.sort(v).values), (
        "shuffle_vec changed the SET of values — it should only reorder them."
    )
    assert abs(shuffled.norm().item() - v.norm().item()) < 1e-4, (
        "shuffle_vec changed the vector's norm — it should be scale-preserving."
    )
    assert not torch.equal(shuffled, v), (
        "shuffle_vec returned the vector unchanged — reordering didn't happen."
    )


def test_shuffle_is_deterministic_per_seed_and_varies_across_seeds():
    v = torch.randn(32)
    a = shuffle_vec(v, random.Random(0))
    b = shuffle_vec(v, random.Random(0))
    c = shuffle_vec(v, random.Random(1))

    assert torch.equal(a, b), "Same seed should give the same shuffle."
    assert not torch.equal(a, c), "Different seeds should (almost always) differ."