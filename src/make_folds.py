import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold


def assign_folds(df, n_splits=5, seed=0):
    d = df[df.role == "dev_defect"].copy()
    key = d.label.apply(lambda s: s if "," not in s else "multi")
    d["fold"] = -1
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    col = d.columns.get_loc("fold")
    for k, (_, te) in enumerate(skf.split(d, key)):
        d.iloc[te, col] = k
    return d[["image", "label", "fold"]]


def main():
    board = sys.argv[1] if len(sys.argv) > 1 else "pcb1"
    df = pd.read_csv(Path("eval") / f"split_{board}.csv")
    out = assign_folds(df)
    out.to_csv(Path("eval") / f"folds_{board}.csv", index=False)
    print(out.groupby(["fold", "label"]).size().to_string())


if __name__ == "__main__":
    main()