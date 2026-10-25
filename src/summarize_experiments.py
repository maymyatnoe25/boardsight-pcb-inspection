import json
import sys
from pathlib import Path

import numpy as np

from src.cnn_cv import paired_bootstrap


def load(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def arrays(r):
    return {k: np.array(r[k]) for k in ("cnn_def", "cnn_norm", "an_def", "an_norm")}


def main():
    board = sys.argv[1] if len(sys.argv) > 1 else "pcb1"
    p2, p3 = Path(f"eval/e2_runs_{board}.jsonl"), Path(f"eval/e3_runs_{board}.jsonl")
    if p2.exists():
        print("E2: defect type held out of CNN training")
        print(f"{'type':10s} {'train def':>9s} {'test def':>8s} {'CNN':>6s} {'anomaly':>8s} "
              f"{'diff':>7s}  95% interval")
        for r in load(p2):
            lo, hi = paired_bootstrap([arrays(r)], n=500)
            print(f"{r['name'].split(chr(39))[1]:10s} {r['n_train_defects']:9d} "
                  f"{r['n_test_defects']:8d} {r['cnn_auc']:6.3f} {r['anomaly_auc']:8.3f} "
                  f"{r['cnn_auc'] - r['anomaly_auc']:+7.3f}  {lo:+.3f} to {hi:+.3f}")
    if p3.exists():
        groups = {}
        for r in load(p3):
            groups.setdefault(r["key"].split(":")[2], []).append(r)
        order = sorted(groups, key=lambda s: (s == "all", 0 if s == "all" else int(s)))
        print("\nE3: learning curve (mean over the folds that have finished)")
        print(f"{'defects':>8s} {'folds':>5s} {'CNN':>6s} {'anomaly':>8s} {'diff':>7s}  95% interval")
        for n in order:
            rs = groups[n]
            c = np.mean([r["cnn_auc"] for r in rs])
            a = np.mean([r["anomaly_auc"] for r in rs])
            lo, hi = paired_bootstrap([arrays(r) for r in rs], n=500)
            used = int(np.mean([r["n_train_defects"] for r in rs]))
            print(f"{n + ' (' + str(used) + ')':>8s} {len(rs):5d} {c:6.3f} {a:8.3f} "
                  f"{c - a:+7.3f}  {lo:+.3f} to {hi:+.3f}")


if __name__ == "__main__":
    main()