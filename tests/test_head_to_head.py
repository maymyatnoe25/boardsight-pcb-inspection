import numpy as np

from src.head_to_head import fa_at_recall, miss_at_fa


def test_false_alarms_at_90_percent_recall():
    d = np.arange(1, 11, dtype=float)          # 10 defects, scores 1..10
    n = np.array([0.5, 2.5, 3.5, 8.5, 0.1])
    # 90% of 10 = 9 defects, so the threshold is the 9th highest score (2)
    assert fa_at_recall(d, n, 0.9) == 3 / 5


def test_missed_defects_at_5_percent_false_alarms():
    n = np.arange(100, dtype=float)
    d = np.array([50.0, 97.0, 99.5, 120.0])
    assert miss_at_fa(d, n, 0.05) == 1 / 4