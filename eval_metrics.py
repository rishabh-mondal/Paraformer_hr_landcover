"""Class unification and scoring shared by evaluate.py and rescore.py (no torch needed).

Mappings to the 4 evaluation classes 0 built-up, 1 tree canopy, 2 low vegetation, 3 water; 255 = ignored.
  "paper"  : Table S1 of the Paraformer supplementary material (CVPR 2024 supp. / arXiv 2403.02746v3, p. 15)
             NLCD 31 barren land -> low vegetation; CCLC 4 barren -> built-up.
  "legacy" : our first mapping, barren ignored on both sides (kept only to reproduce old numbers).
Merging is argmax-first: the most likely of the 17 NLCD-coded outputs is taken, then mapped (the paper does not say).
"""
import numpy as np

IGNORE = 255
UNIFIED = ["built-up", "tree canopy", "low vegetation", "water"]
LABEL_CLASSES = [0, 11, 12, 21, 22, 23, 24, 31, 41, 42, 43, 52, 71, 81, 82, 90, 95]   # prediction index -> NLCD code (= utils.LABEL_CLASSES)
N_GT = 16                                                                             # CCLC raw values 0..15 (0 / 15 = no data)

_NLCD_COMMON = {11: 3, 21: 0, 22: 0, 23: 0, 24: 0, 41: 1, 42: 1, 43: 1, 90: 1, 52: 2, 71: 2, 81: 2, 82: 2, 95: 2}
_CCLC_COMMON = {1: 3, 2: 1, 3: 2, 5: 0, 6: 0}
MAPPINGS = {
    "paper":  ({**_NLCD_COMMON, 31: 2}, {**_CCLC_COMMON, 4: 0}),
    "legacy": (dict(_NLCD_COMMON),       dict(_CCLC_COMMON)),
}

def luts(mapping):
    """(prediction-index LUT of length 17, CCLC LUT of length 256), values 0..3 or 255."""
    nlcd, cclc = MAPPINGS[mapping]
    pred = np.array([nlcd.get(c, IGNORE) for c in LABEL_CLASSES], dtype=np.uint8)
    gt = np.full(256, IGNORE, dtype=np.uint8)
    for k, v in cclc.items(): gt[k] = v
    return pred, gt

def raw_counts(pred_idx, gt_raw):
    """Full-resolution joint histogram: rows = prediction index 0..16, cols = CCLC raw value 0..15."""
    g = np.minimum(gt_raw.astype(np.int64), N_GT - 1)
    return np.bincount(pred_idx.astype(np.int64).ravel() * N_GT + g.ravel(), minlength=17 * N_GT).reshape(17, N_GT)

def unify(raw, mapping):
    """Collapse a 17 x 16 raw table into the 4 x 4 confusion matrix (rows = ground truth, cols = prediction)."""
    p_lut, g_lut = luts(mapping)
    cm = np.zeros((4, 4), dtype=np.int64)
    for i in range(17):
        for j in range(N_GT):
            p, g = p_lut[i], g_lut[j]
            if p != IGNORE and g != IGNORE: cm[g, p] += raw[i, j]
    return cm

def scores(cm):
    tp = np.diag(cm); iou = tp / np.maximum(cm.sum(0) + cm.sum(1) - tp, 1)
    return {"mIoU": float(iou.mean() * 100), "IoU": {c: float(v * 100) for c, v in zip(UNIFIED, iou)},
            "OA": float(tp.sum() / max(cm.sum(), 1) * 100), "pixels": int(cm.sum())}

def summarize(states, tiles, raws, mapping):
    """Per-state / pooled results + per-tile rows from stacked raw tables."""
    cms, per_tile = {}, []
    for st, t, raw in zip(states, tiles, raws):
        cm = unify(raw, mapping); cms[st] = cms.get(st, 0) + cm
        s = scores(cm); per_tile.append({"state": st, "tile": t, "mIoU": s["mIoU"], "OA": s["OA"]})
    res = {"mapping": mapping, "per_state": {st: scores(cm) for st, cm in sorted(cms.items())}}
    res["mean_over_states_mIoU"] = float(np.mean([v["mIoU"] for v in res["per_state"].values()]))
    res["pooled"] = scores(sum(cms.values()))
    return res, per_tile
