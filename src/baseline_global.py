import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torchvision.models import ResNet18_Weights, resnet18

from src.visa_data import ROOT

SIZE = (288, 384)  # height, width
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
ROLES = ("train_normal", "val_normal", "dev_defect")  # test roles are never loaded


def load_tensor(rel):
    img = Image.open(ROOT / rel).convert("RGB").resize((SIZE[1], SIZE[0]), Image.BILINEAR)
    x = (np.asarray(img, dtype=np.float32) / 255 - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1))


def extract(model, rels, batch=16):
    feats, timing = [], []
    with torch.no_grad():
        for i in range(0, len(rels), batch):
            x = torch.stack([load_tensor(r) for r in rels[i:i + batch]])
            t = time.perf_counter()
            f = model(x)
            timing.append((time.perf_counter() - t, len(x)))
            feats.append(f.numpy())
            print(f"  {min(i + batch, len(rels))}/{len(rels)}", end="\r", flush=True)
    return np.concatenate(feats), timing


def main():
    board = sys.argv[1] if len(sys.argv) > 1 else "pcb1"
    df = pd.read_csv(Path("eval") / f"split_{board}.csv")
    df = df[df.role.isin(ROLES)].reset_index(drop=True)
    assert not df.role.str.startswith("test").any()

    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    model.eval()

    cache = Path("data/cache")
    cache.mkdir(parents=True, exist_ok=True)
    cf = cache / f"baseline_{board}_{SIZE[0]}x{SIZE[1]}.npz"
    if cf.exists() and list(np.load(cf, allow_pickle=True)["rels"]) == df.image.tolist():
        feats = np.load(cf, allow_pickle=True)["feats"]
        timing = None
    else:
        feats, timing = extract(model, df.image.tolist())
        np.savez(cf, rels=np.array(df.image.tolist(), dtype=object), feats=feats)
        print()

    train = feats[(df.role == "train_normal").to_numpy()]
    val_mask = df.role.isin(["val_normal", "dev_defect"]).to_numpy()
    val_df, val_f = df[val_mask].reset_index(drop=True), feats[val_mask]
    d = torch.cdist(torch.from_numpy(val_f), torch.from_numpy(train))
    score = d.min(dim=1).values.numpy()

    y = val_df.role.eq("dev_defect").to_numpy().astype(int)
    print(f"\nvalidation images: {len(val_df)} ({y.sum()} defective, {len(y) - y.sum()} normal)")
    print(f"image-level AUROC (all defects): {roc_auc_score(y, score):.3f}")
    for t in ("melt", "missing", "bent", "scratch"):
        has = val_df.label.str.contains(t).to_numpy()
        keep = has | (y == 0)
        print(f"  AUROC for {t:8s} ({int(has.sum()):2d} images): "
              f"{roc_auc_score(has[keep].astype(int), score[keep]):.3f}")
    if timing and len(timing) > 1:
        sec = sum(s for s, _ in timing[1:]) / sum(n for _, n in timing[1:])
        print(f"model time per image on this CPU (first batch excluded): {sec * 1000:.0f} ms")


if __name__ == "__main__":
    main()