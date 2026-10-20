import numpy as np
import torch

from src.metrics import bootstrap_auc_ci
from src.patch_model import build_bank, patch_scores


def test_patch_scores_match_bruteforce():
    g = torch.Generator().manual_seed(0)
    q, b = torch.randn(50, 8, generator=g), torch.randn(300, 8, generator=g)
    assert torch.allclose(patch_scores(q, b, chunk=64),
                          torch.cdist(q, b).min(dim=1).values, atol=1e-4)


def test_bank_size_and_determinism():
    feats = np.random.default_rng(0).random((5, 4, 6, 7)).astype(np.float16)
    a, b = build_bank(feats, 0.2, seed=1), build_bank(feats, 0.2, seed=1)
    assert a.shape == (int(5 * 6 * 7 * 0.2), 4)
    assert torch.equal(a, b)
    assert build_bank(feats, 1.0).shape == (210, 4)


def test_bootstrap_interval_for_perfect_scores():
    y = np.array([0] * 20 + [1] * 10)
    s = np.array([0.0] * 20 + [1.0] * 10)
    assert bootstrap_auc_ci(y, s, n=50) == (1.0, 1.0)