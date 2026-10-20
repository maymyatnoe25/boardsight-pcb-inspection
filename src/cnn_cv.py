import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from torchvision.models import ResNet18_Weights, resnet18

from src.visa_data import ROOT

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
ROLES = ("train_normal", "val_normal", "dev_defect")  # test roles are never loaded
SIZE = (288, 384)  # height, width


def load_images(rels, board, use_cache):
    cache = Path("data/cache")
    cache.mkdir(parents=True, exist_ok=True)
    f = cache / f"cnn_{board}_{SIZE[0]}x{SIZE[1]}.npy"
    keyf = f.with_suffix(".txt")
    if use_cache and f.exists() and keyf.exists() and keyf.read_text().split("\n") == rels:
        return torch.from_numpy(np.load(f))
    h, w = SIZE
    arr = np.zeros((len(rels), 3, h, w), dtype=np.uint8)
    for i, r in enumerate(rels):
        img = Image.open(ROOT / r).convert("RGB").resize((w, h), Image.BILINEAR)
        arr[i] = np.asarray(img).transpose(2, 0, 1)
        print(f"  images {i + 1}/{len(rels)}", end="\r", flush=True)
    print()
    if use_cache:
        np.save(f, arr)
        keyf.write_text("\n".join(rels))
    return torch.from_numpy(arr)


def augment(xb, g, pad=8):
    n, c, h, w = xb.shape
    xp = F.pad(xb.float(), (pad, pad, pad, pad), mode="reflect")
    out = torch.empty(n, c, h, w)
    for i in range(n):
        dy, dx = torch.randint(0, 2 * pad + 1, (2,), generator=g).tolist()
        out[i] = xp[i, :, dy:dy + h, dx:dx + w]
    gain = 1 + (torch.rand(n, 1, 1, 1, generator=g) - 0.5) * 0.2
    bias = (torch.rand(n, 1, 1, 1, generator=g) - 0.5) * 20
    return (out * gain + bias).clamp(0, 255)


def make_model():
    m = resnet18(weights=ResNet18_Weights.DEFAULT)
    m.fc = nn.Linear(512, 1)
    return m


def train_fold(imgs, y, tr_idx, epochs, seed, bs=16, lr=1e-4):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    model = make_model()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    ytr = y[tr_idx]
    pos_weight = torch.tensor(float((ytr == 0).sum() / max(1, (ytr == 1).sum())))
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    for ep in range(epochs):
        model.train()
        perm = tr_idx[torch.randperm(len(tr_idx), generator=g).numpy()]
        t, total = time.perf_counter(), 0.0
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            x = (augment(imgs[torch.as_tensor(b)], g) / 255 - MEAN) / STD
            out = model(x).squeeze(1)
            loss = lossf(out, torch.from_numpy(y[b]).float())
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        print(f"    epoch {ep + 1}/{epochs}  loss {total / len(perm):.3f}  "
              f"({time.perf_counter() - t:.0f} s)", flush=True)
    return model


@torch.no_grad()
def predict(model, imgs, idx, bs=32):
    model.eval()
    out = []
    for i in range(0, len(idx), bs):
        x = (imgs[torch.as_tensor(idx[i:i + bs])].float() / 255 - MEAN) / STD
        out.append(model(x).squeeze(1))
    return torch.cat(out).numpy()


def auc(pos, neg):
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    return float(roc_auc_score(y, np.r_[pos, neg]))


def paired_bootstrap(folds, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n):
        d = []
        for f in folds:
            i = rng.integers(0, len(f["cnn_def"]), len(f["cnn_def"]))
            j = rng.integers(0, len(f["cnn_norm"]), len(f["cnn_norm"]))
            d.append(auc(f["cnn_def"][i], f["cnn_norm"][j]) - auc(f["an_def"][i], f["an_norm"][j]))
        means.append(np.mean(d))
    return [float(v) for v in np.percentile(means, [2.5, 97.5])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("board", nargs="?", default="pcb1")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: N images per role")
    args = ap.parse_args()

    df = pd.read_csv(Path("eval") / f"split_{args.board}.csv")
    df = df[df.role.isin(ROLES)]
    folds = pd.read_csv(Path("eval") / f"folds_{args.board}.csv")[["image", "fold"]]
    df = df.merge(folds, on="image", how="left")
    if args.limit:
        df = df.groupby("role").head(args.limit)
        args.folds = args.folds.split(",")[0]
    df = df.reset_index(drop=True)
    assert not df.role.str.startswith("test").any()

    anom = pd.read_csv(Path("eval") / f"scores_anomaly_{args.board}_val.csv").set_index("image").score
    imgs = load_images(df.image.tolist(), args.board, use_cache=not args.limit)
    y = (df.role == "dev_defect").to_numpy().astype(int)
    val_norm = np.where(df.role == "val_normal")[0]

    results = []
    for k in [int(v) for v in args.folds.split(",")]:
        te_def = np.where((df.role == "dev_defect") & (df.fold == k))[0]
        tr = np.where((df.role == "train_normal") | ((df.role == "dev_defect") & (df.fold != k)))[0]
        print(f"\nfold {k}: train {len(tr)} images ({int(y[tr].sum())} defective), "
              f"score {len(te_def)} defective + {len(val_norm)} normal", flush=True)
        t0 = time.perf_counter()
        model = train_fold(imgs, y, tr, args.epochs, seed=k)
        cd, cn = predict(model, imgs, te_def), predict(model, imgs, val_norm)
        ad, an = anom.loc[df.image.iloc[te_def]].to_numpy(), anom.loc[df.image.iloc[val_norm]].to_numpy()
        r = dict(fold=k, n_def=len(te_def), cnn_auc=auc(cd, cn), anomaly_auc=auc(ad, an),
                 cnn_def=cd, cnn_norm=cn, an_def=ad, an_norm=an)
        results.append(r)
        print(f"  CNN AUROC {r['cnn_auc']:.3f} | anomaly detector AUROC on the same images "
              f"{r['anomaly_auc']:.3f} | {time.perf_counter() - t0:.0f} s for this fold", flush=True)

    c = np.array([r["cnn_auc"] for r in results])
    a = np.array([r["anomaly_auc"] for r in results])
    print(f"\nmean over {len(results)} fold(s): CNN {c.mean():.3f} | anomaly {a.mean():.3f}")
    if args.limit:
        print("Smoke test only: nothing was saved.")
        return
    lo, hi = paired_bootstrap(results)
    print(f"mean difference (CNN minus anomaly): {c.mean() - a.mean():+.3f}  "
          f"(95% paired bootstrap interval {lo:+.3f} to {hi:+.3f})")
    out = dict(board=args.board, epochs=args.epochs, folds=[r["fold"] for r in results],
               cnn_auc=c.tolist(), anomaly_auc=a.tolist(), diff_mean=float(c.mean() - a.mean()),
               diff_ci=[lo, hi])
    Path(f"eval/cnn_cv_{args.board}.json").write_text(json.dumps(out, indent=1))
    print(f"Saved eval/cnn_cv_{args.board}.json")


if __name__ == "__main__":
    main()