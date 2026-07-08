import pytest
from attn_phase.stats import mann_whitney_test

def test_direction_solved_greater_than_failed():
    # Solved values consistently higher
    solved = [10.0, 11.0, 12.0, 13.0, 14.0]
    failed = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = mann_whitney_test(solved, failed, metric="test_metric")
    assert result.effect_size_r < 0
    assert result.direction == "solved > failed"

def test_direction_solved_less_than_failed():
    # Solved values consistently lower
    solved = [1.0, 2.0, 3.0, 4.0, 5.0]
    failed = [10.0, 11.0, 12.0, 13.0, 14.0]
    result = mann_whitney_test(solved, failed, metric="test_metric")
    assert result.effect_size_r > 0
    assert result.direction == "solved < failed"

def test_direction_no_effect():
    # Identical values, or no significant difference
    solved = [1.0, 2.0, 3.0, 4.0, 5.0]
    failed = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = mann_whitney_test(solved, failed, metric="test_metric")
    assert not result.significant
    assert result.direction == "no effect"

def test_direction_insufficient_data():
    # Less than 2 values in one group
    solved = [1.0]
    failed = [1.0, 2.0, 3.0]
    result = mann_whitney_test(solved, failed, metric="test_metric")
    assert result.direction == "insufficient data"
    assert not result.significant
    
def test_bonferroni_correction():
    # p_corrected should be p_raw * n_comparisons, capped at 1.0
    solved = [10.0, 11.0, 12.0, 13.0, 14.0]
    failed = [1.0, 2.0, 3.0, 4.0, 5.0]
    
    # 2 comparisons
    res2 = mann_whitney_test(solved, failed, metric="test", n_comparisons=2)
    assert res2.p_corrected == min(1.0, res2.p_value * 2)
    
    # Huge number of comparisons to test cap at 1.0
    res_large = mann_whitney_test(solved, failed, metric="test", n_comparisons=1000000)
    assert res_large.p_corrected == 1.0
