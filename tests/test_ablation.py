import numpy as np

from src.anomaly_ablation import pixel_auroc, pointing_hit


def test_pointing_hit_inside_and_outside():
    m = np.zeros((20, 20))
    m[5, 5] = 1
    mask = np.zeros((20, 20), bool)
    mask[4:7, 4:7] = True
    assert pointing_hit(m, mask, slack=0)
    m2 = np.zeros((20, 20))
    m2[15, 15] = 1
    assert not pointing_hit(m2, mask, slack=2)


def test_pixel_auroc_perfect_map():
    m = np.zeros((10, 10))
    m[2:4, 2:4] = 1
    mask = np.zeros((10, 10), bool)
    mask[2:4, 2:4] = True
    assert pixel_auroc([m, np.zeros((10, 10))], [mask, None], neg_frac=1.0) == 1.0