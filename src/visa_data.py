from pathlib import Path

import pandas as pd

ROOT = Path(r"C:\datasets\VisA")


def _find(cols, word):
    for c in cols:
        if word in c.lower():
            return c
    raise KeyError(f"no column containing '{word}' in {list(cols)}")


def _resolve(board, rel):
    for base in (ROOT, ROOT / board):
        p = base / str(rel)
        if p.exists():
            return p
    return None


def load_board(board="pcb1"):
    """One row per image. Paths are stored relative to ROOT, so they work on any machine."""
    df = pd.read_csv(ROOT / board / "image_anno.csv")
    img_c, lab_c = _find(df.columns, "image"), _find(df.columns, "label")
    try:
        mask_c = _find(df.columns, "mask")
    except KeyError:
        mask_c = None
    rows = []
    for _, r in df.iterrows():
        label = str(r[lab_c]).strip().lower()
        ip = _resolve(board, r[img_c])
        mp = _resolve(board, r[mask_c]) if mask_c and isinstance(r[mask_c], str) else None
        rows.append({
            "image": ip.relative_to(ROOT).as_posix() if ip else None,
            "mask": mp.relative_to(ROOT).as_posix() if mp else None,
            "label": label,
            "is_defect": label != "normal",
            "types": [] if label == "normal" else label.split(","),
        })
    return pd.DataFrame(rows)