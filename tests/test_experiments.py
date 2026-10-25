import numpy as np
import pandas as pd

from src.cnn_experiments import e2_split, e3_split


def fake():
    rows = [{"role": "train_normal", "label": "normal", "fold": np.nan} for _ in range(10)]
    labels = ["melt"] * 6 + ["bent"] * 4 + ["melt,scratch"] * 2
    rows += [{"role": "dev_defect", "label": lab, "fold": i % 3} for i, lab in enumerate(labels)]
    rows += [{"role": "val_normal", "label": "normal", "fold": np.nan} for _ in range(5)]
    return pd.DataFrame(rows)


def test_e2_removes_every_image_with_the_held_out_type():
    df = fake()
    tr, te = e2_split(df, "melt")
    assert set(df.label.iloc[te]) == {"melt", "melt,scratch"}
    assert not any("melt" in lab for lab in df.label.iloc[tr])
    assert (df.role.iloc[tr] == "train_normal").sum() == 10
    assert set(tr).isdisjoint(te)


def test_e3_never_trains_on_the_test_fold_and_respects_n():
    df = fake()
    for k in range(3):
        tr, te = e3_split(df, k, "3")
        assert set(tr).isdisjoint(te)
        assert (df.role.iloc[tr] == "dev_defect").sum() == 3
        tr_all, _ = e3_split(df, k, "all")
        assert (df.role.iloc[tr_all] == "dev_defect").sum() == 8