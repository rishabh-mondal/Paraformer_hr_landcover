"""Side-by-side test-set figures for one epoch of the three Paraformer runs.

usage: /opt/miniconda3/envs/rishabh_sat/bin/python make_test_figures.py 50
Needs evaluate.py to have been run with --save_png 4 for each run (eval/test_epoch<E>/png + per_tile.csv).
Writes runs/compare_epoch<E>/tiles/<state>_<tile>.jpg (every test tile) and
       runs/compare_epoch<E>/overview_one_tile_per_state.png (the median tile of each state).
Panels: NAIP RGB | CCLC ground truth | 3 model predictions, all in the 4 unified evaluation classes (paper Table S1), at 1/4 resolution.
"""
import os, sys
import numpy as np, pandas as pd, rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from PIL import Image

plt.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "stix"})
EP = int(sys.argv[1]) if len(sys.argv) > 1 else 50
F = 4                                           # decimation, must match evaluate.py --save_png
RUNS = [("Baseline: ViT-B/16 fine-tuned", "paraformer_adamw_lr1e-4_w128_bs8"),
        ("DINOv3 satellite, frozen", "paraformer_dinov3_sat_frozen_adamw_lr1e-4_w128_bs8"),
        ("DINOv3 web, frozen", "paraformer_dinov3_web_frozen_adamw_lr1e-4_w128_bs8")]
STATES = {"de": "Delaware", "ny": "New York", "md": "Maryland", "pa": "Pennsylvania", "va": "Virginia", "wv": "West Virginia"}
CLASSES = ["Built-up", "Tree canopy", "Low vegetation", "Water"]
COLORS = ["#9c9c9c", "#267300", "#a3ff73", "#00c5ff"]
CMAP = ListedColormap(COLORS + ["#ffffff"]); NORM = BoundaryNorm([0, 1, 2, 3, 4, 256], CMAP.N)
CCLC_LUT = np.full(256, 255, np.uint8); CCLC_LUT[[1, 2, 3, 4, 5, 6]] = [3, 1, 2, 0, 0, 0]   # paper Table S1: barren -> built-up
OUT = f"runs/compare_epoch{EP}"; INK, MUTED = "#0b0b0b", "#52514e"

def load_scores():
    """Per-tile mIoU/OA per run: per_tile.csv when the pass is finished, otherwise parsed from the live eval log."""
    import re
    per = {}
    for lab, run in RUNS:
        f = f"runs/{run}/eval/test_epoch{EP}/per_tile.csv"
        if os.path.exists(f):
            per[run] = pd.read_csv(f).set_index("tile"); continue
        rows = []
        log = f"runs/{run}/eval/test_epoch{EP}.log"
        if os.path.exists(log):
            for l in open(log):
                m = re.search(r"\] (\w\w) (\S+)\s+mIoU=([0-9.]+)\s+OA=([0-9.]+)", l)
                if m: rows.append({"tile": m.group(2), "state": m.group(1), "mIoU": float(m.group(3)), "OA": float(m.group(4))})
        per[run] = pd.DataFrame(rows).set_index("tile") if rows else pd.DataFrame()
    return per

def tile_arrays(row):
    with rasterio.open(row.image_fn) as f: rgb = np.moveaxis(f.read([1, 2, 3]), 0, -1)[::F, ::F]
    with rasterio.open(row.gt_fn) as f: gt = CCLC_LUT[f.read(1)[::F, ::F]]
    preds = []
    for lab, run in RUNS:
        p = f"runs/{run}/eval/test_epoch{EP}/png/" + os.path.basename(row.image_fn).replace("_naip-new.tif", "_pred4.png")
        preds.append(np.array(Image.open(p)) if os.path.exists(p) else None)
    return rgb, gt, preds

def draw_row(axes, row, per, label_rows=True, fs=13):
    tile = os.path.basename(row.image_fn); st = os.path.basename(os.path.dirname(row.image_fn))[:2]
    rgb, gt, preds = tile_arrays(row)
    axes[0].imshow(rgb); axes[1].imshow(gt, cmap=CMAP, norm=NORM, interpolation="nearest")
    titles = ["NAIP RGB (1 m)", "CCLC ground truth"]
    for ax, (lab, run), p in zip(axes[2:], RUNS, preds):
        if p is not None: ax.imshow(p, cmap=CMAP, norm=NORM, interpolation="nearest")
        s = per[run].loc[tile] if tile in per[run].index else None
        titles.append(f"{lab}\nmIoU {s.mIoU:.1f}  ·  OA {s.OA:.1f}" if s is not None else f"{lab}\n(not scored yet)")
    for ax, t in zip(axes, titles):
        if label_rows: ax.set_title(t, fontsize=fs, color=INK)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_color("#c8c7c2")
    axes[0].set_ylabel(f"{STATES[st]}\n{tile.replace('_naip-new.tif', '')}", fontsize=fs, color=INK)
    return st, tile

def legend(fig, y):
    fig.legend([Patch(color=c) for c in COLORS] + [Patch(facecolor="white", edgecolor="#999")], CLASSES + ["Ignored (no data)"],
               loc="center", bbox_to_anchor=(0.5, y), ncol=5, frameon=False, fontsize=14, labelcolor=INK)

if __name__ == "__main__":
    os.makedirs(f"{OUT}/tiles", exist_ok=True)
    df = pd.read_csv("dataset/CSV_list/Chesapeake_all_test.csv"); per = load_scores()
    only_done = "--all" not in sys.argv
    done = set.intersection(*[set(per[r].index) for _, r in RUNS]) if all(len(per[r]) for _, r in RUNS) else set()
    n = 0
    for _, row in df.iterrows():
        tile = os.path.basename(row.image_fn)
        if only_done and tile not in done: continue
        fig, axes = plt.subplots(1, 5, figsize=(22, 6.4))
        st, _ = draw_row(axes, row, per)
        fig.suptitle(f"Test tile, epoch {EP} checkpoints: {STATES[st]}", fontsize=18, weight="bold", color=INK, y=0.99)
        legend(fig, 0.04); fig.subplots_adjust(left=0.05, right=0.99, top=0.82, bottom=0.1, wspace=0.04)
        fig.savefig(f"{OUT}/tiles/{st}_{tile.replace('_naip-new.tif', '')}.jpg", dpi=90, pil_kwargs={"quality": 90}); plt.close(fig); n += 1
    # overview: per state, the tile whose mean mIoU over the three models is the state median
    pick = []
    for st in STATES:
        cand = [t for t in done if f"/{st}/" in df.set_index(df.image_fn.map(os.path.basename)).loc[t].image_fn]
        if not cand: continue
        m = {t: np.mean([per[r].loc[t].mIoU for _, r in RUNS]) for t in cand}
        med = np.median(list(m.values())); pick.append(min(cand, key=lambda t: abs(m[t] - med)))
    if pick:
        fig, axes = plt.subplots(len(pick), 5, figsize=(22, 4.6 * len(pick) + 1.6), squeeze=False)
        rows = df.set_index(df.image_fn.map(os.path.basename))
        for i, t in enumerate(pick):
            draw_row(axes[i], rows.loc[t], per, label_rows=True, fs=12)
        fig.suptitle(f"Test set, epoch {EP}: one representative tile per state (median of that state's tiles)",
                     fontsize=20, weight="bold", color=INK, y=0.995)
        legend(fig, 0.008); fig.subplots_adjust(left=0.06, right=0.99, top=0.965, bottom=0.025, hspace=0.32, wspace=0.04)
        fig.savefig(f"{OUT}/overview_one_tile_per_state.png", dpi=80); plt.close(fig)
    print(f"{n} tile figures in {OUT}/tiles, overview over {len(pick)} states")
