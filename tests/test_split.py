import pandas as pd

from src.make_split import make_split


def fake():
    rows = [{"image": f"n{i}", "mask": None, "label": "normal", "is_defect": False,
             "types": []} for i in range(100)]
    for kind in ("melt", "bent"):
        rows += [{"image": f"{kind}{i}", "mask": "m", "label": kind, "is_defect": True,
                  "types": [kind]} for i in range(20)]
    return pd.DataFrame(rows)


KW = dict(seed=0, n_val_normal=20, n_test_normal=20, n_test_defect=10)


def test_every_image_has_exactly_one_role_and_sizes_match():
    out = make_split(fake(), **KW)
    assert (out.role != "").all()
    assert out.role.value_counts().to_dict() == {
        "train_normal": 60, "val_normal": 20, "test_normal": 20,
        "dev_defect": 30, "test_defect": 10}


def test_split_is_deterministic_and_stratified():
    a, b = make_split(fake(), **KW), make_split(fake(), **KW)
    assert a.role.tolist() == b.role.tolist()
    t = a[a.role == "test_defect"].label.value_counts()
    assert t["melt"] == 5 and t["bent"] == 5