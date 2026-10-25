import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.anomaly_ablation import full_res, get_feats, load_mask, pointing_hit
from src.cnn_cv import MEAN, ROLES, SIZE, STD, load_images, train_fold
from src.patch_model import PatchExtractor, build_bank, score_maps

LAYERS = ("layer3", "layer4")
AN_SIZE = (384, 512)


def gradcam(model, x, layer_name):
    """Grad-CAM maps (N x h x w) for the single output logit."""
    store = {}

    def hook(_, __, out):
        out.retain_grad()
        store["a"] = out

    handle = getattr(model, layer_name).register_forward_hook(hook)
    model.eval()
    model.zero_grad()
    with torch.enable_grad():
        model(x).squeeze(1).sum().backward()
    handle.remove()
    a = store["a"]
    w = a.grad.mean(dim=(2, 3), keepdim=True)
    return F.relu((w * a).sum(dim=1)).detach().numpy()


def pointing_rows(maps_full, rels, mask_rels, size):
    slack = round(0.05 * size[0])
    rows = []
    for m, rel, mr in zip(maps_full, rels, mask_rels):
        mask = load_mask(mr, size)
        rows.append(dict(image=rel, hit5=pointing_hit(m, mask, slack),
                         hit0=pointing_hit(m, mask, 0)))
    return rows


def append(path, obj):
    with path.open("a") as f:
        f.write(json.dumps(obj) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("board", nargs="?", default="pcb1")
    args = ap.parse_args()

    df = pd.read_csv(Path("eval") / f"split_{args.board}.csv")
    df = df[df.role.isin(ROLES)]
    folds = pd.read_csv(Path("eval") / f"folds_{args.board}.csv")[["image", "fold"]]
    df = df.merge(folds, on="image", how="left").reset_index(drop=True)
    assert not df.role.str.startswith("test").any()

    out = Path(f"eval/e4_{args.board}.jsonl")
    done = set()
    if out.exists():
        done = {json.loads(line)["fold"] for line in out.read_text().splitlines() if line.strip()}
    y = (df.role == "dev_defect").to_numpy().astype(int)
    dev = np.where(df.role == "dev_defect")[0]

    if -1 not in done:
        print("anomaly heatmaps for the 70 development defects", flush=True)
        ex = PatchExtractor(AN_SIZE)
        feats = get_feats(ex, df.image.tolist(), args.board)
        bank = build_bank(feats[(df.role == "train_normal").to_numpy()], 0.05, 0)
        maps = full_res(score_maps(feats[dev], bank), AN_SIZE)
        rows = pointing_rows(maps, df.image.iloc[dev].tolist(), df["mask"].iloc[dev].tolist(), AN_SIZE)
        for r in rows:
            r["method"] = "anomaly"
        append(out, dict(fold=-1, rows=rows))
        del feats, bank, maps

    imgs = load_images(df.image.tolist(), args.board, use_cache=True)
    for k in range(5):
        if k in done:
            continue
        tr = np.where((df.role == "train_normal") | ((df.role == "dev_defect") & (df.fold != k)))[0]
        te = np.where((df.role == "dev_defect") & (df.fold == k))[0]
        print(f"\nfold {k}: training on {len(tr)} images, explaining {len(te)} held-out defects", flush=True)
        model = train_fold(imgs, y, tr, 6, seed=k)
        x = (imgs[torch.as_tensor(te)].float() / 255 - MEAN) / STD
        rows = []
        for layer in LAYERS:
            maps = full_res(gradcam(model, x, layer), SIZE)
            part = pointing_rows(maps, df.image.iloc[te].tolist(), df["mask"].iloc[te].tolist(), SIZE)
            for r in part:
                r["method"] = f"gradcam_{layer}"
            rows += part
        append(out, dict(fold=k, rows=rows))
        print(f"  fold {k} saved", flush=True)


if __name__ == "__main__":
    main()