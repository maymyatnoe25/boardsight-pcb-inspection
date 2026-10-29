import numpy as np

from src.score_test_set import paired_auc_diff_ci, rate_with_ci


def test_exact_interval_for_zero_of_ten():
    p, lo, hi = rate_with_ci(0, 10)
    assert p == 0 and lo < 1e-9 and 0.30 < hi < 0.32


def test_paired_difference_is_zero_for_identical_scores():
    y = np.array([0] * 20 + [1] * 10)
    s = np.r_[np.linspace(0, 1, 20), np.linspace(0.5, 1.5, 10)]
    lo, hi = paired_auc_diff_ci(y, s, s, n=100)
    assert abs(lo) < 1e-9 and abs(hi) < 1e-9