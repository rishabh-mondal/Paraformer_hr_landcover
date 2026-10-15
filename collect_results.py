"""Combine every Table S1 test result of the three runs into one CSV: runs/test_results_tableS1.csv
(one row per model x epoch; Average = mean of the six state mIoUs = the paper's Table 1 metric)."""
import os, pandas as pd
RUNS = [("Baseline: ImageNet ViT-B/16, fine-tuned", "paraformer_adamw_lr1e-4_w128_bs8"),
        ("DINOv3 ViT-L/16 satellite, frozen", "paraformer_dinov3_sat_frozen_adamw_lr1e-4_w128_bs8"),
        ("DINOv3 ViT-L/16 web, frozen", "paraformer_dinov3_web_frozen_adamw_lr1e-4_w128_bs8")]
OUT = "runs/test_results_tableS1.csv"
STATES = {"de": "DE", "ny": "NY", "md": "MD", "pa": "PA", "va": "VA", "wv": "WV"}
CLASSES = ["built-up", "tree_canopy", "low_vegetation", "water"]
def collect():
    rows = []
    for model, run in RUNS:
        f = f"runs/{run}/eval/history_test.csv"
        if not os.path.exists(f): continue
        h = pd.read_csv(f)
        if "mapping" not in h: continue
        for _, r in h[h.mapping == "paper"].sort_values("epoch").iterrows():
            row = {"model": model, "run": run, "epoch": int(r.epoch), "average_mIoU": r.avg_mIoU,
                   "pooled_mIoU": r.pooled_mIoU, "pooled_OA": r.pooled_OA}
            row.update({f"{lab}_mIoU": r[f"{s}_mIoU"] for s, lab in STATES.items()})
            row.update({f"IoU_{c}": r[f"pooled_{c}"] for c in CLASSES})
            rows.append(row)
    df = pd.DataFrame(rows)
    if len(df): df = df.round(2)
    df.to_csv(OUT, index=False)
    return df
if __name__ == "__main__":
    df = collect(); print(f"{OUT}: {len(df)} rows"); print(df[["model", "epoch", "average_mIoU", "pooled_mIoU"]].to_string(index=False) if len(df) else "")
