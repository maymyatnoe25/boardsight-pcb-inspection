import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.anomaly_ablation import get_feats
from src.anomaly_eval import gallery
from src.patch_model import PatchExtractor, build_bank, score_maps

ROLES = ("train_normal", "val_normal", "dev_defect")  # test roles are never loaded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("board", nargs="?", default="pcb1")
    ap.add_argument("--size", default="384,512", help="height,width")
    ap.add_argument("--fraction", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    size = tuple(int(v) for v in args.size.split(","))

    df = pd.read_csv(Path("eval") / f"split_{args.board}.csv")
    df = df[df.role.isin(ROLES)].reset_index(drop=True)
    assert not df.role.str.startswith("test").any()

    ex = PatchExtractor(size)
    feats = get_feats(ex, df.image.tolist(), args.board)
    bank = build_bank(feats[(df.role == "train_normal").to_numpy()], args.fraction, args.seed)
    print(f"bank: {len(bank)} patches")

    vi = np.where(df.role.isin(["val_normal", "dev_defect"]))[0]
    vdf = df.iloc[vi].reset_index(drop=True)
    maps = score_maps(feats[vi], bank)
    scores = maps.reshape(len(vdf), -1).max(1)

    out = vdf[["image", "role", "label"]].copy()
    out["score"] = scores
    out.to_csv(Path("eval") / f"scores_anomaly_{args.board}_final_val.csv", index=False)
    print("saved", len(out), "scores")

    y = vdf.role.eq("dev_defect").to_numpy().astype(int)
    vmin, vmax = float(np.median(maps[y == 0])), float(maps.max())
    d, n_ = np.where(y == 1)[0], np.where(y == 0)[0]
    Path("data").mkdir(exist_ok=True)
    gallery("data/final_examples.png", d[:6], vdf, maps, scores, size, vmin, vmax,
            "First six defective validation boards (outline = true defect)")
    gallery("data/final_missed.png", d[np.argsort(scores[d])[:6]], vdf, maps, scores, size,
            vmin, vmax, "Six lowest-scoring defective boards (most likely missed)")
    gallery("data/final_false_alarms.png", n_[np.argsort(-scores[n_])[:6]], vdf, maps, scores,
            size, vmin, vmax, "Six highest-scoring normal boards (worst false alarms)")
    print("saved three galleries in data/")


if __name__ == "__main__":
    main()