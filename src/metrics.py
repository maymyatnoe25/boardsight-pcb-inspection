import numpy as np
from sklearn.metrics import roc_auc_score


def bootstrap_auc_ci(y, s, n=1000, seed=0, alpha=0.05):
    """95% bootstrap interval for AUROC, resampling defective and normal images separately."""
    y, s = np.asarray(y), np.asarray(s)
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    aucs = []
    for _ in range(n):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        aucs.append(roc_auc_score(y[idx], s[idx]))
    return (float(np.percentile(aucs, 100 * alpha / 2)),
            float(np.percentile(aucs, 100 * (1 - alpha / 2))))