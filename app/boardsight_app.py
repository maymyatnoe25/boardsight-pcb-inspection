import json
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.stats import binomtest

from src import patch_model
from src.cnn_cv import MEAN, SIZE, STD, make_model
from src.patch_model import PatchExtractor, patch_scores
from src.visa_data import ROOT

BOARD, AN_SIZE = "pcb1", (384, 512)

st.set_page_config(page_title="BoardSight", layout="wide")


@st.cache_resource(show_spinner="Loading the models...")
def load_models():
    ex = PatchExtractor(AN_SIZE)
    bank = torch.load(f"data/cache/bank_{BOARD}_384x512_0.05.pt")
    cnn = make_model()
    cnn.load_state_dict(torch.load(f"data/cache/cnn_final_{BOARD}.pt"))
    cnn.eval()
    th = json.loads(Path(f"eval/thresholds_final_{BOARD}.json").read_text())
    return ex, bank, cnn, th


def analyse(img, ex, bank, cnn):
    h, w = AN_SIZE
    t0 = time.perf_counter()
    a = img.convert("RGB").resize((w, h), Image.BILINEAR)
    x = (np.asarray(a, dtype=np.float32) / 255 - patch_model.MEAN) / patch_model.STD
    x = torch.from_numpy(x.transpose(2, 0, 1))[None]
    with torch.no_grad():
        f = ex.features(x)[0]
        c, fh, fw = f.shape
        d = patch_scores(f.reshape(c, -1).T.float(), bank).reshape(fh, fw).numpy()
    t_an = time.perf_counter() - t0

    t0 = time.perf_counter()
    hh, ww = SIZE
    b = img.convert("RGB").resize((ww, hh), Image.BILINEAR)
    xc = torch.from_numpy(np.asarray(b).transpose(2, 0, 1).copy())[None].float() / 255
    with torch.no_grad():
        s_cnn = float(cnn((xc - MEAN) / STD).squeeze())
    t_cnn = time.perf_counter() - t0
    return d, float(d.max()), s_cnn, t_an, t_cnn


def show_maps(img, d, t_an, mask_rel):
    H, W = AN_SIZE
    base = img.convert("RGB").resize((W, H))
    m = F.interpolate(torch.from_numpy(d)[None, None], size=(H, W), mode="bilinear",
                      align_corners=False)[0, 0].numpy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].imshow(base)
    axes[0].set_title("Image")
    axes[1].imshow(base)
    axes[1].imshow(m, cmap="jet", alpha=0.5, vmin=0.6 * t_an, vmax=max(1.2 * t_an, float(m.max())))
    axes[1].set_title("Anomaly heatmap (red = least like a good board)")
    if mask_rel:
        mk = np.array(Image.open(ROOT / mask_rel).resize((W, H), Image.NEAREST))
        mk = (mk[..., 0] if mk.ndim == 3 else mk) > 0
        if mk.any():
            axes[0].contour(mk, levels=[0.5], colors="red", linewidths=1)
            axes[1].contour(mk, levels=[0.5], colors="white", linewidths=1)
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    return fig


def test_table():
    p = Path(f"eval/test_results_{BOARD}.json")
    if not p.exists():
        return None
    r = json.loads(p.read_text())
    rows = []
    for name, key in (("Anomaly detector", "anomaly detector"), ("Fine-tuned CNN", "CNN")):
        m = r[key]
        c = binomtest(m["caught"], 30).proportion_ci(confidence_level=0.95, method="exact")
        f = binomtest(m["false_alarms"], 200).proportion_ci(confidence_level=0.95, method="exact")
        lat = r["latency_ms"]["anomaly" if key.startswith("anomaly") else "cnn"]
        rows.append({"Model": name,
                     "Defects caught (of 30)": f"{m['caught']} ({c.low:.0%} to {c.high:.0%})",
                     "False alarms (of 200 good boards)": f"{m['false_alarms']} ({f.low:.1%} to {f.high:.1%})",
                     "AUROC": f"{m['auroc']:.3f} ({m['auroc_ci'][0]:.3f} to {m['auroc_ci'][1]:.3f})",
                     "Time per image (ms)": round(lat)})
    return pd.DataFrame(rows), r


st.title("BoardSight")
st.caption("Flags possibly defective circuit boards and shows where to look. Decision support for "
           "a human inspector, not an automatic verdict.")
st.warning("Trained on one board type only (VisA PCB1). Other boards, cameras or lighting are "
           "outside what was tested, and results there are unknown.")

ex, bank, cnn, th = load_models()
split = pd.read_csv(Path("eval") / f"split_{BOARD}.csv")
samples = pd.concat([split[split.role == "dev_defect"].head(15),
                     split[split.role == "val_normal"].head(15)]).sample(frac=1, random_state=0)
samples = samples.reset_index(drop=True)

choice = st.radio("Image source", ["Sample board", "Upload a photo"], horizontal=True)
img, mask_rel, truth = None, None, None
if choice == "Sample board":
    k = st.selectbox("Choose a sample", range(len(samples)), format_func=lambda i: f"Sample {i + 1}")
    row = samples.iloc[k]
    img = Image.open(ROOT / row["image"])
    mask_rel = row["mask"] if isinstance(row["mask"], str) else None
    truth = row["label"]
else:
    up = st.file_uploader("Photo of a circuit board", type=["jpg", "jpeg", "png"])
    if up:
        img = Image.open(up)

if img is not None:
    with st.spinner("Inspecting..."):
        d, s_an, s_cnn, t_an, t_cnn = analyse(img, ex, bank, cnn)
    flag_an, flag_cnn = s_an > th["anomaly"], s_cnn > th["cnn"]

    if flag_an and flag_cnn:
        st.error("Flag for human review: both models flag this board.")
    elif flag_an or flag_cnn:
        who = "the anomaly detector" if flag_an else "the CNN"
        st.warning(f"Flag for human review: only {who} flags this board. The models disagree.")
    else:
        st.info("Not flagged by either model. This does not guarantee the board has no defect.")

    st.pyplot(show_maps(img, d, th["anomaly"], mask_rel))
    c1, c2 = st.columns(2)
    c1.metric("Anomaly score", f"{s_an:.2f}", f"flag above {th['anomaly']:.2f}", delta_color="off")
    c1.caption(f"Computed in {t_an * 1000:.0f} ms. A distance, not a probability.")
    c2.metric("CNN score", f"{s_cnn:.2f}", f"flag above {th['cnn']:.2f}", delta_color="off")
    c2.caption(f"Computed in {t_cnn * 1000:.0f} ms. A raw model output, not a probability.")
    if truth is not None:
        st.caption(f"Ground truth for this sample (demo only): {truth}. "
                   "The heatmap outlines the true defect when there is one.")

st.divider()
st.subheader("How well did it work on boards it had never seen?")
tt = test_table()
if tt is None:
    st.info("Test results file not found.")
else:
    table, r = tt
    st.dataframe(table, hide_index=True)
    lo, hi = r["auroc_diff_ci"]
    st.markdown(
        f"The AUROC difference between the models is {r['auroc_diff']:+.3f} (95% interval "
        f"{lo:+.3f} to {hi:+.3f}), so there is no clear difference in ranking quality. The CNN "
        "raised more false alarms than the anomaly detector, so the catch counts are not at the "
        "same false-alarm rate.")

with st.expander("Limitations"):
    st.markdown("""
- One board type (PCB1 of the VisA dataset), 30 defective and 200 good test boards. One
  training seed per model. Intervals are wide.
- Thresholds were set so that about 5% of good validation boards are flagged. The anomaly score
  and the CNN output are not probabilities.
- Missed defects tend to be tiny. The anomaly heatmap pointed at the true defect for 96% of
  development defects. A Grad-CAM map for the CNN pointed at the defect for only 27 to 43%,
  so it is not shown.
- False alarms can come from background marks and unusual board positions.
- Dataset: VisA (Zou et al., 2022). Images are not redistributed.
""")