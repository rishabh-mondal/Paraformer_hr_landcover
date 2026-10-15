"""Append one evaluate.py results.json to a history CSV (one row per checkpoint).
usage: python append_history.py <results.json> <epoch> <history.csv>"""
import sys, json, os, pandas as pd
res, epoch, out = json.load(open(sys.argv[1])), int(sys.argv[2]), sys.argv[3]
row = {"epoch": epoch, "mapping": res.get("mapping", "legacy"), "avg_mIoU": res["mean_over_states_mIoU"], "pooled_mIoU": res["pooled"]["mIoU"], "pooled_OA": res["pooled"]["OA"]}
for st, v in res["per_state"].items():
    row[f"{st}_mIoU"] = v["mIoU"]
    for c, x in v["IoU"].items(): row[f"{st}_{c.replace(' ', '_')}"] = x
for c, x in res["pooled"]["IoU"].items(): row[f"pooled_{c.replace(' ', '_')}"] = x
df = pd.DataFrame([row])
if os.path.exists(out):
    old = pd.read_csv(out)
    if "mapping" not in old: old.insert(1, "mapping", "legacy")
    keep = ~((old.epoch == epoch) & (old.mapping == row["mapping"]))
    df = pd.concat([old[keep], df]).sort_values(["epoch", "mapping"])
df.round(2).to_csv(out, index=False); print(f"{out}: {len(df)} rows, epochs {df.epoch.tolist()}")
