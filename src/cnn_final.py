import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.cnn_cv import ROLES, load_images, predict, train_fold


def main():
    board = sys.argv[1] if len(sys.argv) > 1 else "pcb1"
    df = pd.read_csv(Path("eval") / f"split_{board}.csv")
    df = df[df.role.isin(ROLES)].reset_index(drop=True)
    assert not df.role.str.startswith("test").any()
    imgs = load_images(df.image.tolist(), board, use_cache=True)
    y = (df.role == "dev_defect").to_numpy().astype(int)
    tr = np.where(df.role.isin(["train_normal", "dev_defect"]))[0]
    print(f"training the final CNN on {len(tr)} images ({int(y[tr].sum())} defective)", flush=True)
    model = train_fold(imgs, y, tr, 6, seed=0)
    Path("data/cache").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), f"data/cache/cnn_final_{board}.pt")

    val_norm = np.where(df.role == "val_normal")[0]
    cnn_scores = predict(model, imgs, val_norm)
    an = pd.read_csv(Path("eval") / f"scores_anomaly_{board}_final_val.csv")
    an_norm = an[an.role == "val_normal"].score.to_numpy()
    an_dev = an[an.role == "dev_defect"].score.to_numpy()
    t_cnn, t_an = float(np.percentile(cnn_scores, 95)), float(np.percentile(an_norm, 95))
    print(f"CNN threshold (95th percentile of {len(cnn_scores)} validation normals): {t_cnn:.3f}"
          f" -> flags {(cnn_scores > t_cnn).sum()} of them")
    print(f"anomaly threshold (95th percentile of {len(an_norm)} validation normals): {t_an:.3f}"
          f" -> flags {(an_norm > t_an).sum()} of them; catches "
          f"{(an_dev > t_an).sum()} of {len(an_dev)} development defects")
    Path(f"eval/thresholds_final_{board}.json").write_text(json.dumps(dict(
        rule="95th percentile of validation-normal scores; flag if strictly above",
        cnn=t_cnn, anomaly=t_an), indent=1))
    print(f"saved eval/thresholds_final_{board}.json")


if __name__ == "__main__":
    main()