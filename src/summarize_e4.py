import json
import sys
from pathlib import Path

import numpy as np


def ci(v, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    v = np.asarray(v, float)
    m = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    board = sys.argv[1] if len(sys.argv) > 1 else "pcb1"
    rows = []
    for line in Path(f"eval/e4_{board}.jsonl").read_text().splitlines():
        if line.strip():
            rows += json.loads(line)["rows"]
    by = {}
    for r in rows:
        by.setdefault(r["method"], {})[r["image"]] = r
    base = by["anomaly"]
    print(f"{'method':18s} {'n':>3s} {'hit (5% slack)':>26s} {'strict hit':>11s} "
          f"{'minus anomaly (5% slack)':>34s}")
    for m in sorted(by, key=lambda s: (s != "anomaly", s)):
        common = sorted(set(by[m]) & set(base))
        h = np.array([by[m][i]["hit5"] for i in common], float)
        s = np.array([by[m][i]["hit0"] for i in common], float)
        lo, hi = ci(h)
        txt = ""
        if m != "anomaly":
            b = np.array([base[i]["hit5"] for i in common], float)
            dl, dh = ci(h - b)
            txt = f"{h.mean() - b.mean():+.2f} ({dl:+.2f} to {dh:+.2f})"
        print(f"{m:18s} {len(common):3d} {h.mean():8.2f} ({lo:.2f} to {hi:.2f}) "
              f"{s.mean():11.2f} {txt:>34s}")


if __name__ == "__main__":
    main()