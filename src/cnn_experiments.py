import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.cnn_cv import ROLES, auc, load_images, predict, train_fold

TYPES = ("melt", "missing", "bent", "scratch")


def e2_split(df, t):
    is_def = (df.role == "dev_defect").to_numpy()
    has = df.label.apply(lambda s: t in s.split(",")).to_numpy()
    tr = np.where((df.role == "train_normal").to_numpy() | (is_def & ~has))[0]
    te = np.where(is_def & has)[0]
    return tr, te


def e3_split(df, k, n):
    is_def = (df.role == "dev_defect").to_numpy()
    fold = df.fold.to_numpy()
    te = np.where(is_def & (fold == k))[0]
    pool = np.where(is_def & (fold != k))[0]
    if n != "all":
        rng = np.random.default_rng(1000 + 10 * k + int(n))
        pool = rng.choice(pool, size=min(int(n), len(pool)), replace=False)
    tr = np.concatenate([np.where((df.role == "train_normal").to_numpy())[0], pool])
    return tr, te


def done_keys(path):
    if not path.exists():
        return set()
    return {json.loads(line)["key"] for line in path.read_text().splitlines() if line.strip()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", choices=["e2", "e3"])
    ap.add_argument("board", nargs="?", default="pcb1")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--sizes", default="5,10,25,all")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: N images per role")
    args = ap.parse_args()

    df = pd.read_csv(Path("eval") / f"split_{args.board}.csv")
    df = df[df.role.isin(ROLES)]
    folds = pd.read_csv(Path("eval") / f"folds_{args.board}.csv")[["image", "fold"]]
    df = df.merge(folds, on="image", how="left")
    if args.limit:
        df = df.groupby("role").head(args.limit)
    df = df.reset_index(drop=True)
    assert not df.role.str.startswith("test").any()

    anom = pd.read_csv(Path("eval") / f"scores_anomaly_{args.board}_val.csv").set_index("image").score
    imgs = load_images(df.image.tolist(), args.board, use_cache=not args.limit)
    y = (df.role == "dev_defect").to_numpy().astype(int)
    val_norm = np.where(df.role == "val_normal")[0]

    out = Path(f"eval/{args.experiment}_runs_{args.board}.jsonl")
    done = set() if args.limit else done_keys(out)

    if args.experiment == "e2":
        jobs = [(f"e2:{t}", f"hold out '{t}'", e2_split(df, t), 100 + i)
                for i, t in enumerate(TYPES)]
    else:
        jobs = [(f"e3:{k}:{n}", f"fold {k}, {n} defects", e3_split(df, k, n), k)
                for k in [int(v) for v in args.folds.split(",")]
                for n in args.sizes.split(",")]

    for key, name, (tr, te), seed in jobs:
        if key in done or len(te) == 0:
            continue
        print(f"\n{name}: train {len(tr)} images ({int(y[tr].sum())} defective), "
              f"score {len(te)} defective + {len(val_norm)} normal", flush=True)
        model = train_fold(imgs, y, tr, args.epochs, seed)
        cd, cn = predict(model, imgs, te), predict(model, imgs, val_norm)
        ad = anom.loc[df.image.iloc[te]].to_numpy()
        an = anom.loc[df.image.iloc[val_norm]].to_numpy()
        row = dict(key=key, name=name, n_train_defects=int(y[tr].sum()),
                   n_test_defects=len(te), cnn_auc=auc(cd, cn), anomaly_auc=auc(ad, an),
                   cnn_def=cd.tolist(), cnn_norm=cn.tolist(), an_def=ad.tolist(),
                   an_norm=an.tolist(), def_images=df.image.iloc[te].tolist())
        print(f"  CNN AUROC {row['cnn_auc']:.3f} | anomaly detector {row['anomaly_auc']:.3f}",
              flush=True)
        if not args.limit:
            with out.open("a") as f:
                f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()