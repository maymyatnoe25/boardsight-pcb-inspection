import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.cnn_cv import paired_bootstrap
from src.head_to_head import auc


def main():
    board = sys.argv[1] if len(sys.argv) > 1 else "pcb1"
    split = pd.read_csv(Path("eval") / f"split_{board}.csv")
    norm_imgs = split[split.role == "val_normal"].image.tolist()
    fpath = Path("eval") / f"scores_anomaly_{board}_final_val.csv"
    final = pd.read_csv(fpath).set_index("image").score
    an_norm = final.loc[norm_imgs].to_numpy()

    groups = {}
    for line in Path(f"eval/e3_runs_{board}.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            groups.setdefault(r["key"].split(":")[2], []).append(r)
    order = sorted(groups, key=lambda s: (s == "all", 0 if s == "all" else int(s)))

    print("E3 against the FINAL anomaly detector (384x512, 5% bank)")
    print(f"{'defects':>8s} {'folds':>5s} {'CNN':>6s} {'anomaly':>8s} {'diff':>7s}  95% interval")
    for n in order:
        folds = [dict(cnn_def=np.array(r["cnn_def"]), cnn_norm=np.array(r["cnn_norm"]),
                      an_def=final.loc[r["def_images"]].to_numpy(), an_norm=an_norm)
                 for r in groups[n]]
        c = np.mean([auc(f["cnn_def"], f["cnn_norm"]) for f in folds])
        a = np.mean([auc(f["an_def"], f["an_norm"]) for f in folds])
        lo, hi = paired_bootstrap(folds, n=500)
        print(f"{n:>8s} {len(folds):5d} {c:6.3f} {a:8.3f} {c - a:+7.3f}  {lo:+.3f} to {hi:+.3f}")


if __name__ == "__main__":
    main()