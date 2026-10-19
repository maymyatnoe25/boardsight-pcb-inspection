import sys
from collections import Counter

import numpy as np
from PIL import Image

from src.visa_data import ROOT, load_board


def main():
    board = sys.argv[1] if len(sys.argv) > 1 else "pcb1"
    df = load_board(board)
    d = df[df.is_defect]
    print("rows:", len(df), "| normal:", int((~df.is_defect).sum()),
          "| defective:", len(d))
    print("image files not found:", int(df.image.isna().sum()))
    print("defective images without a mask:", int(d["mask"].isna().sum()))
    print("normal images with a mask:", int(df[~df.is_defect]["mask"].notna().sum()))
    print("images containing each type:", dict(Counter(t for ts in d.types for t in ts)))
    print("single-type:", int((d.types.apply(len) == 1).sum()),
          "| multi-type:", int((d.types.apply(len) > 1).sum()))
    print("image sizes (width, height):",
          dict(Counter(Image.open(ROOT / p).size for p in df.image.dropna())))

    areas, bad_shape = [], 0
    for _, r in d.dropna(subset=["mask"]).iterrows():
        m = np.array(Image.open(ROOT / r["mask"]))
        if m.ndim == 3:
            m = m[..., 0]
        if m.shape[::-1] != Image.open(ROOT / r["image"]).size:
            bad_shape += 1
        areas.append(float((m > 0).mean()) * 100)
    if areas:
        q = np.percentile(areas, [0, 25, 50, 75, 100])
        print("defect area as % of the image (min, 25%, median, 75%, max):",
              [round(x, 3) for x in q])
        print("empty masks:", sum(1 for a in areas if a == 0),
              "| masks with a different size than the image:", bad_shape)


if __name__ == "__main__":
    main()