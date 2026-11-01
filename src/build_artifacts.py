from pathlib import Path

import pandas as pd
import torch

from src.anomaly_ablation import get_feats
from src.patch_model import PatchExtractor, build_bank

BOARD, SIZE = "pcb1", (384, 512)
ROLES = ("train_normal", "val_normal", "dev_defect")


def main():
    df = pd.read_csv(Path("eval") / f"split_{BOARD}.csv")
    df = df[df.role.isin(ROLES)].reset_index(drop=True)
    ex = PatchExtractor(SIZE)
    feats = get_feats(ex, df.image.tolist(), BOARD)
    bank = build_bank(feats[(df.role == "train_normal").to_numpy()], 0.05, 0)
    del feats
    Path("data/cache").mkdir(parents=True, exist_ok=True)
    torch.save(bank, f"data/cache/bank_{BOARD}_384x512_0.05.pt")
    print("saved memory bank with", len(bank), "patches")


if __name__ == "__main__":
    main()