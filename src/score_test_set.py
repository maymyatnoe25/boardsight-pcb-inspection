import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score

from src.anomaly_ablation import get_feats
from src.anomaly_eval import gallery
from src.cnn_cv import MEAN, SIZE, STD, load_images, make_model, predict
from src.metrics import bootstrap_auc_ci
from src.patch_model import PatchExtractor, build_bank, patch_scores, score_maps
from src.visa_data import ROOT

BOARD = "pcb1"
AN_SIZE = (384, 512)
TRAIN_ROLES = ("train_normal", "val_normal", "dev_defect")


def rate_with_ci(k, n):
    ci = binomtest(int(k), int(n)).proportion_ci(confidence_level=0.95, method="exact")
    return k / n, float(ci.low), float(ci.high)


def paired_auc_diff_ci(y, a, b, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    y, a, b = np.asarray(y), np.asarray(a), np.asarray(b)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    d = []
    for _ in range(n):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        d.append(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx]))
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def latency_ms(fn, rels, n=20, warmup=3):
    times = []
    for r in rels[:n + warmup]:
        t = time.perf_counter()
        fn(r)
        times.append(time.perf_counter() - t)
    return 1000 * float(np.mean(times[warmup:]))


def git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()


def main():
    out_json = Path(f"eval/test_results_{BOARD}.json")
    if out_json.exists():
        raise SystemExit("The test set has already been scored once (results file exists). "
                         "It must not be scored again.")
    if git("status", "--porcelain"):
        raise SystemExit("Uncommitted changes. Commit and push first, so the plan is recorded "
                         "before the test run.")
    commit = git("rev-parse", "HEAD")

    split = pd.read_csv(Path("eval") / f"split_{BOARD}.csv")
    tr_df = split[split.role.isin(TRAIN_ROLES)].reset_index(drop=True)
    test_df = split[split.role.isin(["test_normal", "test_defect"])].reset_index(drop=True)
    y = (test_df.role == "test_defect").to_numpy().astype(int)
    assert y.sum() == 30 and (1 - y).sum() == 200, "unexpected test set size"
    th = json.loads(Path(f"eval/thresholds_final_{BOARD}.json").read_text())
    t_cnn, t_an = th["cnn"], th["anomaly"]
    t_old = json.loads(Path(f"eval/thresholds_{BOARD}.json").read_text())["anomaly"]

    print("anomaly detector: building the memory bank and scoring", flush=True)
    ex = PatchExtractor(AN_SIZE)
    feats = get_feats(ex, tr_df.image.tolist(), BOARD)
    bank = build_bank(feats[(tr_df.role == "train_normal").to_numpy()], 0.05, 0)
    del feats
    maps = score_maps(ex.extract_all(test_df.image.tolist()), bank)
    an = maps.reshape(len(test_df), -1).max(1)

    print("CNN: scoring", flush=True)
    model = make_model()
    model.load_state_dict(torch.load(f"data/cache/cnn_final_{BOARD}.pt"))
    model.eval()
    imgs = load_images(test_df.image.tolist(), BOARD, use_cache=False)
    cnn = predict(model, imgs, np.arange(len(test_df)))

    def an_one(r):
        f = ex.features(ex.load(r)[None])[0]
        patch_scores(f.reshape(f.shape[0], -1).T.float(), bank)

    def cnn_one(r):
        h, w = SIZE
        img = Image.open(ROOT / r).convert("RGB").resize((w, h), Image.BILINEAR)
        x = torch.from_numpy(np.asarray(img).transpose(2, 0, 1).copy())[None].float() / 255
        with torch.no_grad():
            model((x - MEAN) / STD)

    rels = test_df.image.tolist()
    lat = {"anomaly": latency_ms(an_one, rels), "cnn": latency_ms(cnn_one, rels)}

    results = {"commit": commit, "thresholds": {"cnn": t_cnn, "anomaly": t_an, "anomaly_old": t_old}}
    print(f"\ncode version (git commit): {commit[:10]}")
    for name, s, t in (("anomaly detector", an, t_an), ("CNN", cnn, t_cnn),
                       ("anomaly at old 90%-recall threshold", an, t_old)):
        caught = int((s[y == 1] > t).sum())
        alarms = int((s[y == 0] > t).sum())
        cr, clo, chi = rate_with_ci(caught, 30)
        fr, flo, fhi = rate_with_ci(alarms, 200)
        print(f"{name:38s} threshold {t:7.3f} | caught {caught}/30 = {cr:.0%} "
              f"({clo:.0%} to {chi:.0%}) | false alarms {alarms}/200 = {fr:.1%} "
              f"({flo:.1%} to {fhi:.1%})")
        results[name] = dict(threshold=t, caught=caught, false_alarms=alarms)

    for name, s in (("anomaly detector", an), ("CNN", cnn)):
        lo, hi = bootstrap_auc_ci(y, s)
        auc = float(roc_auc_score(y, s))
        print(f"AUROC {name:17s}: {auc:.3f} (95% bootstrap interval {lo:.3f} to {hi:.3f})")
        results[name].update(auroc=auc, auroc_ci=[lo, hi])
    dlo, dhi = paired_auc_diff_ci(y, cnn, an)
    diff = float(roc_auc_score(y, cnn) - roc_auc_score(y, an))
    print(f"AUROC difference, CNN minus anomaly: {diff:+.3f} "
          f"(95% paired bootstrap interval {dlo:+.3f} to {dhi:+.3f})")
    print(f"latency per image on this CPU, loading included: anomaly {lat['anomaly']:.0f} ms, "
          f"CNN {lat['cnn']:.0f} ms")
    results.update(auroc_diff=diff, auroc_diff_ci=[dlo, dhi], latency_ms=lat)

    out = test_df[["image", "label", "role"]].copy()
    out["anomaly_score"], out["cnn_score"] = an, cnn
    out.to_csv(f"eval/test_scores_{BOARD}.csv", index=False)

    d, n_ = np.where(y == 1)[0], np.where(y == 0)[0]
    vmin, vmax = float(np.median(maps[y == 0])), float(maps.max())
    missed = d[an[d] <= t_an]
    missed = missed[np.argsort(an[missed])][:6]
    alarmed = n_[an[n_] > t_an]
    alarmed = alarmed[np.argsort(-an[alarmed])][:6]
    Path("data").mkdir(exist_ok=True)
    if len(missed):
        gallery("data/test_missed.png", missed, test_df, maps, an, AN_SIZE, vmin, vmax,
                "Test defects missed by the anomaly detector (lowest scores first)")
    if len(alarmed):
        gallery("data/test_false_alarms.png", alarmed, test_df, maps, an, AN_SIZE, vmin, vmax,
                "Test normals flagged by the anomaly detector (highest scores first)")
    out_json.write_text(json.dumps(results, indent=1))
    print(f"\nsaved {out_json}, eval/test_scores_{BOARD}.csv and the galleries in data/")


if __name__ == "__main__":
    main()