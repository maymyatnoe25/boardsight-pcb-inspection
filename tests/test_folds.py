import pandas as pd

from src.make_folds import assign_folds


def test_every_defect_in_exactly_one_fold():
    rows = [{"image": f"{t}{i}", "label": t, "role": "dev_defect"}
            for t in ("melt", "bent", "melt,scratch") for i in range(10)]
    rows += [{"image": f"n{i}", "label": "normal", "role": "train_normal"} for i in range(20)]
    out = assign_folds(pd.DataFrame(rows))
    assert len(out) == 30 and out.image.is_unique
    assert set(out.fold) == {0, 1, 2, 3, 4}
    assert out.groupby("fold").size().between(5, 7).all()