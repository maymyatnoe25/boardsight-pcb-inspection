import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.cnn_cv import paired_bootstrap


def auc(d, n):
    y = np.r_[np.ones(len(d)), np.zeros(len(n))]
    return float(roc_auc_score(y, np.r_[d, n]))


def threshold_at_recall(d, recall=0.9):
    k = int(np.ceil(round(recall * len(d), 9)))
    return float(np.sort(d)[::-1][k - 1])


def fa_at_recall(d, n, recall=0.9):
    return float((n >= threshold_at_recall(d, recall)).mean())


def miss_at_fa(d, n, fa=0.05):
    return float((d <= np.quantile(n, 1 - fa)).mean())


def main():
    board = sys.argv[1] if len(sys.argv) > 1 else "pcb1"
    split = pd.read_csv(Path("eval") / f"split_{board}.csv")
    norm_imgs = split[split.role == "val_normal"].image.tolist()
    fdf = pd.read_csv(Path("eval") / f"scores_anomaly_{board}_final_val.csv")
    final = fdf.set_index("image").score
    an_norm = final.loc[norm_imgs].to_numpy()

    lines = Path(f"eval/e3_runs_{board}.jsonl").read_text().splitlines()
    runs = [json.loads(line) for line in lines if line.strip()]
    runs = sorted((r for r in runs if r["key"].endswith(":all")), key=lambda r: r["key"])
    assert len(runs) == 5, f"expected 5 'all' runs, found {len(runs)}"

    folds = []
    for r in runs:
        assert len(r["cnn_norm"]) == len(norm_imgs)
        folds.append(dict(cnn_def=np.array(r["cnn_def"]), cnn_norm=np.array(r["cnn_norm"]),
                          an_def=final.loc[r["def_images"]].to_numpy(), an_norm=an_norm,
                          old_def=np.array(r["an_def"]), old_norm=np.array(r["an_norm"])))

    models = {"CNN, 56 labels, 288x384": ("cnn_def", "cnn_norm"),
              "anomaly, default 288x384:0.05": ("old_def", "old_norm"),
              "anomaly, final 384x512:0.05": ("an_def", "an_norm")}
    print(f"{'model':32s} {'AUROC':>6s} {'false alarms at 90% recall':>30s} "
          f"{'defects missed at 5% false alarms':>36s}")
    for name, (kd, kn) in models.items():
        a = [auc(f[kd], f[kn]) for f in folds]
        fa = [fa_at_recall(f[kd], f[kn]) for f in folds]
        ms = [miss_at_fa(f[kd], f[kn]) for f in folds]
        print(f"{name:32s} {np.mean(a):6.3f} "
              f"{np.mean(fa):12.1%} ({min(fa):.1%} to {max(fa):.1%})"
              f"{np.mean(ms):17.1%} ({min(ms):.1%} to {max(ms):.1%})")

    old = [dict(cnn_def=f["cnn_def"], cnn_norm=f["cnn_norm"], an_def=f["old_def"],
                an_norm=f["old_norm"]) for f in folds]
    for label, fs, key in (("CNN minus default anomaly (sanity check, was +0.016)", old, "old"),
                           ("CNN minus final anomaly", folds, "final")):
        diff = np.mean([auc(f["cnn_def"], f["cnn_norm"]) -
                        auc(f["an_def" if key == "final" else "an_def"],
                            f["an_norm"]) for f in fs])
        lo, hi = paired_bootstrap(fs, n=500)
        print(f"\n{label}: {diff:+.3f}  (95% paired bootstrap interval {lo:+.3f} to {hi:+.3f})")

    t_folds = [threshold_at_recall(f["cnn_def"]) for f in folds]
    t_cnn = float(np.mean(t_folds))
    dev = fdf[fdf.role == "dev_defect"].score.to_numpy()
    t_an = threshold_at_recall(dev)
    print(f"\nCNN per-fold thresholds: {[round(t, 2) for t in t_folds]} -> mean {t_cnn:.2f}")
    print(f"anomaly threshold (90% recall on all {len(dev)} development defects): {t_an:.3f}"
          f"  | false-alarm rate on validation normals: {(an_norm >= t_an).mean():.1%}")
    Path(f"eval/thresholds_{board}.json").write_text(json.dumps(
        dict(cnn=t_cnn, cnn_per_fold=t_folds, anomaly=t_an), indent=1))
    print(f"saved eval/thresholds_{board}.json")


if __name__ == "__main__":
    main()