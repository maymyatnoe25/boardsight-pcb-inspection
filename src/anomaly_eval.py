import argparse
import csv
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.metrics import bootstrap_auc_ci
from src.patch_model import PatchExtractor, build_bank, score_maps
from src.visa_data import ROOT

ROLES = ("train_normal", "val_normal", "dev_defect")  # test roles are never loaded


def get_feats(ex, rels, board, limit):
    cache = Path("data/cache")
    cache.mkdir(parents=True, exist_ok=True)
    f = cache / f"patch_{board}_{ex.size[0]}x{ex.size[1]}.npy"
    keyf = f.with_suffix(".txt")
    if not limit and f.exists() and keyf.exists() and keyf.read_text().split("\n") == rels:
        return np.load(f)
    feats = ex.extract_all(rels)
    if not limit:
        np.save(f, feats)
        keyf.write_text("\n".join(rels))
    return feats


def gallery(path, pos, vdf, maps, scores, size, vmin, vmax, title):
    H, W = size
    fig, axes = plt.subplots(2, len(pos), figsize=(4 * len(pos), 7), squeeze=False)
    for j, p in enumerate(pos):
        img = Image.open(ROOT / vdf["image"].iloc[p]).convert("RGB").resize((W, H))
        m = F.interpolate(torch.from_numpy(maps[p])[None, None], size=(H, W),
                          mode="bilinear", align_corners=False)[0, 0].numpy()
        mk = None
        if isinstance(vdf["mask"].iloc[p], str):
            mk = np.array(Image.open(ROOT / vdf["mask"].iloc[p]).resize((W, H), Image.NEAREST))
            mk = (mk[..., 0] if mk.ndim == 3 else mk) > 0
        for r in range(2):
            axes[r, j].imshow(img)
            if r == 1:
                axes[r, j].imshow(m, cmap="jet", alpha=0.5, vmin=vmin, vmax=vmax)
            if mk is not None and mk.any():
                axes[r, j].contour(mk, levels=[0.5], colors="white" if r else "red", linewidths=1)
            axes[r, j].axis("off")
        axes[0, j].set_title(f"{vdf['label'].iloc[p]} | score {scores[p]:.2f}", fontsize=9)
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=70)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("board", nargs="?", default="pcb1")
    ap.add_argument("--size", default="288,384", help="height,width")
    ap.add_argument("--fraction", type=float, default=0.05)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: use N images per role")
    args = ap.parse_args()
    size = tuple(int(v) for v in args.size.split(","))

    df = pd.read_csv(Path("eval") / f"split_{args.board}.csv")
    df = df[df.role.isin(ROLES)]
    if args.limit:
        df = df.groupby("role").head(args.limit)
    df = df.reset_index(drop=True)
    assert not df.role.str.startswith("test").any()

    t0 = time.perf_counter()
    ex = PatchExtractor(size)
    feats = get_feats(ex, df.image.tolist(), args.board, args.limit)
    bank = build_bank(feats[(df.role == "train_normal").to_numpy()], args.fraction)
    print(f"memory bank: {len(bank)} patches, {bank.numel() * 4 / 1e6:.0f} MB")

    vi = np.where(df.role.isin(["val_normal", "dev_defect"]))[0]
    vdf, vf = df.iloc[vi].reset_index(drop=True), feats[vi]
    t = time.perf_counter()
    maps = score_maps(vf, bank)
    sec = (time.perf_counter() - t) / len(vdf)
    scores = maps.reshape(len(vdf), -1).max(1)
    y = vdf.role.eq("dev_defect").to_numpy().astype(int)

    auc = roc_auc_score(y, scores)
    lo, hi = bootstrap_auc_ci(y, scores)
    print(f"\nvalidation: {int(y.sum())} defective, {int((1 - y).sum())} normal")
    print(f"image-level AUROC: {auc:.3f}  (95% bootstrap interval {lo:.3f} to {hi:.3f})")
    for tname in ("melt", "missing", "bent", "scratch"):
        has = vdf.label.str.contains(tname).to_numpy()
        keep = has | (y == 0)
        if has.sum() > 0:
            print(f"  AUROC {tname:8s} ({int(has.sum()):2d} images): "
                  f"{roc_auc_score(has[keep].astype(int), scores[keep]):.3f}")
    print(f"nearest-neighbour search: {sec * 1000:.0f} ms per image (features excluded)")
    print(f"total time {time.perf_counter() - t0:.0f} s")

    if args.limit:
        print("\nSmoke test only: nothing was saved.")
        return
    np.save(f"data/cache/anomaly_{args.board}_val_scores.npy", scores)
    row = dict(model="patch-memory-bank-resnet18", board=args.board, size=args.size,
               fraction=args.fraction, bank_patches=len(bank), auroc=round(auc, 4),
               ci_lo=round(lo, 4), ci_hi=round(hi, 4), search_ms_per_image=round(sec * 1000),
               val_defect=int(y.sum()), val_normal=int((1 - y).sum()))
    path = Path("eval/results_dev.csv")
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)

    vmin, vmax = float(np.median(maps[y == 0])), float(maps.max())
    d, n_ = np.where(y == 1)[0], np.where(y == 0)[0]
    Path("data").mkdir(exist_ok=True)
    gallery("data/gallery_examples.png", d[:6], vdf, maps, scores, size, vmin, vmax,
            "First six defective validation boards (red/white outline = true defect)")
    gallery("data/gallery_missed.png", d[np.argsort(scores[d])[:6]], vdf, maps, scores,
            size, vmin, vmax, "Six lowest-scoring defective boards (most likely missed)")
    gallery("data/gallery_false_alarms.png", n_[np.argsort(-scores[n_])[:6]], vdf, maps,
            scores, size, vmin, vmax, "Six highest-scoring normal boards (worst false alarms)")
    print("Saved three galleries in data/ and a row in eval/results_dev.csv")


if __name__ == "__main__":
    main()