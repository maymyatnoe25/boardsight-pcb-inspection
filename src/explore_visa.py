import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(r"C:\datasets\VisA")


def find_col(cols, word):
    for c in cols:
        if word in c.lower():
            return c
    return None


def resolve(board, rel):
    for base in (ROOT, ROOT / board):
        p = base / str(rel)
        if p.exists():
            return p
    return None


def main():
    board = sys.argv[1] if len(sys.argv) > 1 else "pcb1"
    df = pd.read_csv(ROOT / board / "image_anno.csv")
    print("Rows:", len(df))
    print("Columns:", list(df.columns))
    print(df.head(3).to_string(), "\n")

    for c in df.columns:
        if df[c].nunique() <= 15:
            print(f"--- {c} ---")
            print(df[c].value_counts(dropna=False).to_string(), "\n")

    img_col, mask_col, label_col = (find_col(df.columns, w) for w in ("image", "mask", "label"))
    if not (img_col and label_col):
        print("Could not find image/label columns. Send me the output above.")
        return

    is_bad = df[label_col].astype(str).str.lower().str.contains("anomal")
    bad, good = df[is_bad].head(4), df[~is_bad].head(4)

    first = resolve(board, df[img_col].iloc[0])
    if first is not None:
        print("First image size (width, height):", Image.open(first).size)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, (_, row) in zip(axes[0], bad.iterrows()):
        p = resolve(board, row[img_col])
        if p is None:
            ax.set_title("image not found")
            ax.axis("off")
            continue
        ax.imshow(Image.open(p))
        if mask_col and isinstance(row[mask_col], str):
            m = resolve(board, row[mask_col])
            if m is not None:
                mask = np.array(Image.open(m)) > 0
                ax.imshow(np.ma.masked_where(~mask, mask), cmap="autumn", alpha=0.6)
        ax.set_title("defective (red = mask)")
        ax.axis("off")
    for ax, (_, row) in zip(axes[1], good.iterrows()):
        p = resolve(board, row[img_col])
        if p is not None:
            ax.imshow(Image.open(p))
        ax.set_title("normal")
        ax.axis("off")

    Path("data").mkdir(exist_ok=True)
    out = Path("data") / f"{board}_samples.png"
    plt.tight_layout()
    plt.savefig(out, dpi=80)
    print("Saved", out)


if __name__ == "__main__":
    main()
    