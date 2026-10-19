import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.visa_data import load_board


def make_split(df, seed=0, n_val_normal=200, n_test_normal=200, n_test_defect=30):
    df = df.copy()
    df["role"] = ""
    normal = df[~df.is_defect].sample(frac=1.0, random_state=seed)
    a, b = n_test_normal, n_test_normal + n_val_normal
    df.loc[normal.index[:a], "role"] = "test_normal"
    df.loc[normal.index[a:b], "role"] = "val_normal"
    df.loc[normal.index[b:], "role"] = "train_normal"

    defect = df[df.is_defect]
    key = defect["types"].apply(lambda t: t[0] if len(t) == 1 else "multi")
    counts = key.value_counts()
    key = key.where(key.map(counts) >= 4, "rare")
    test_idx = train_test_split(defect.index.to_numpy(), test_size=n_test_defect,
                                stratify=key.to_numpy(), random_state=seed)[1]
    df.loc[defect.index, "role"] = "dev_defect"
    df.loc[test_idx, "role"] = "test_defect"
    return df


def main():
    board = sys.argv[1] if len(sys.argv) > 1 else "pcb1"
    df = load_board(board)
    assert df.image.notna().all(), "some image files were not found"
    out = make_split(df)
    path = Path("eval") / f"split_{board}.csv"
    out[["image", "mask", "label", "role"]].to_csv(path, index=False)
    print(out.role.value_counts().to_string())
    print("\ndefect labels per role:")
    print(out[out.is_defect].groupby(["role", "label"]).size().to_string())
    print("\nSaved", path)


if __name__ == "__main__":
    main()