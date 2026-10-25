import argparse
import csv
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import binary_dilation
from sklearn.metrics import roc_auc_score

from src.metrics import bootstrap_auc_ci
from src.patch_model import PatchExtractor, build_bank, patch_scores, score_maps
from src.visa_data import ROOT

ROLES = ("train_normal", "val_normal", "dev_defect")  # test roles are never loaded


def full_res(maps, size):
    return F.interpolate(torch.from_numpy(maps)[:, None], size=size, mode="bilinear",
                         align_corners=False)[:, 0].numpy()


def load_mask(rel, size):
    h, w = size
    m = np.array(Image.open(ROOT / rel).resize((w, h), Image.NEAREST))
    return (m[..., 0] if m.ndim == 3 else m) > 0


def pixel_auroc(maps, masks, seed=0, neg_frac=0.02):
    """Pooled pixel AUROC. All defect pixels are kept; negatives are subsampled uniformly."""
    rng = np.random.default_rng(seed)
    ys, ss = [], []
    for m, mk in zip(maps, masks):
        flat = m.ravel()
        pos = np.zeros(flat.shape, bool) if mk is None else mk.ravel()
        neg = np.where(~pos)[0]
        sel = np.concatenate([np.where(pos)[0], neg[rng.random(len(neg)) < neg_frac]])
        ys.append(pos[sel])
        ss.append(flat[sel])
    return float(roc_auc_score(np.concatenate(ys), np.concatenate(ss)))


def pointing_hit(m, mask, slack):
    iy, ix = np.unravel_index(np.argmax(m), m.shape)
    allowed = binary_dilation(mask, iterations=slack) if slack else mask
    return bool(allowed[iy, ix])


def get_feats(ex, rels, board):
    cache = Path("data/cache")
    cache.mkdir(parents=True, exist_ok=True)
    f = cache / f"patch_{board}_{ex.size[0]}x{ex.size[1]}.npy"
    keyf = f.with_suffix(".txt")
    if f.exists() and keyf.exists() and keyf.read_text().split("\n") == rels:
        return np.load(f)
    feats = ex.extract_all(rels)
    np.save(f, feats)
    keyf.write_text("\n".join(rels))
    return feats


def end_to_end_ms(ex, bank, rels, n=15, warmup=3):
    times = []
    for r in rels[:n + warmup]:
        t = time.perf_counter()
        f = ex.features(ex.load(r)[None])[0]
        q = f.reshape(f.shape[0], -1).T.float()
        patch_scores(q, bank)
        times.append(time.perf_counter() - t)
    return 1000 * float(np.mean(times[warmup:]))


def parse(c):
    hw, frac = c.split(":")
    h, w = hw.split("x")
    return (int(h), int(w)), float(frac)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("board")
    ap.add_argument("configs", nargs="+", help="e.g. 288x384:0.05")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(Path("eval") / f"split_{args.board}.csv")
    df = df[df.role.isin(ROLES)].reset_index(drop=True)
    assert not df.role.str.startswith("test").any()
    rels = df.image.tolist()
    out = Path(f"eval/ablation_{args.board}.csv")
    done = set()
    if out.exists():
        old = pd.read_csv(out)
        done = {(r.size, r.fraction, r.seed) for r in old.itertuples()}

    train = (df.role == "train_normal").to_numpy()
    vi = np.where(df.role.isin(["val_normal", "dev_defect"]))[0]
    vdf = df.iloc[vi].reset_index(drop=True)
    y = vdf.role.eq("dev_defect").to_numpy().astype(int)

    for c in args.configs:
        size, frac = parse(c)
        key = (f"{size[0]}x{size[1]}", frac, args.seed)
        if key in done:
            print("skip", c, "seed", args.seed)
            continue
        print(f"\n== {c}  seed {args.seed}", flush=True)
        ex = PatchExtractor(size)
        feats = get_feats(ex, rels, args.board)
        bank = build_bank(feats[train], frac, args.seed)
        t = time.perf_counter()
        maps = score_maps(feats[vi], bank)
        search_ms = 1000 * (time.perf_counter() - t) / len(vdf)
        scores = maps.reshape(len(vdf), -1).max(1)
        auc = roc_auc_score(y, scores)
        lo, hi = bootstrap_auc_ci(y, scores)

        mf = full_res(maps, size)
        masks = [load_mask(m, size) if isinstance(m, str) else None for m in vdf["mask"]]
        px = pixel_auroc(mf, masks, args.seed)
        slack = round(0.05 * size[0])
        hits = [pointing_hit(mf[i], masks[i], slack) for i in np.where(y == 1)[0]]
        e2e = end_to_end_ms(ex, bank, [ROOT.as_posix() and r for r in vdf.image.tolist()])

        row = dict(size=key[0], fraction=frac, seed=args.seed, bank_patches=len(bank),
                   bank_mb=round(bank.numel() * 4 / 1e6), auroc=round(auc, 4),
                   ci_lo=round(lo, 4), ci_hi=round(hi, 4), pixel_auroc=round(px, 4),
                   pointing=round(float(np.mean(hits)), 3), search_ms=round(search_ms),
                   end_to_end_ms=round(e2e))
        print(row, flush=True)
        new = not out.exists()
        with out.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)

    print("\n" + pd.read_csv(out).to_string(index=False))


if __name__ == "__main__":
    main()