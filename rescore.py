"""Re-score an evaluation folder from its raw_counts.npz with any class mapping - no inference needed.
usage: python rescore.py <eval_out_dir> [--mapping paper|legacy] [--write]
Without --write it only prints; with --write it overwrites results.json / per_tile.csv in that folder."""
import argparse, json, os
import numpy as np, pandas as pd
import eval_metrics as em
p = argparse.ArgumentParser(); p.add_argument("out_dir"); p.add_argument("--mapping", default="paper", choices=sorted(em.MAPPINGS))
p.add_argument("--write", action="store_true"); a = p.parse_args()
z = np.load(os.path.join(a.out_dir, "raw_counts.npz"))
res, per_tile = em.summarize(list(z["states"]), list(z["tiles"]), list(z["raws"]), a.mapping)
for st, v in res["per_state"].items(): print(f"{st}: {v['mIoU']:.2f}   " + "  ".join(f"{c}={x:.1f}" for c, x in v["IoU"].items()))
print(f"[{a.mapping}] Average over states: {res['mean_over_states_mIoU']:.2f}   pooled: {res['pooled']['mIoU']:.2f}   OA pooled: {res['pooled']['OA']:.2f}")
if a.write:
    old = json.load(open(os.path.join(a.out_dir, "results.json")))
    json.dump({"model_path": old.get("model_path"), "list_dir": old.get("list_dir"), **res}, open(os.path.join(a.out_dir, "results.json"), "w"), indent=2)
    pd.DataFrame(per_tile).to_csv(os.path.join(a.out_dir, "per_tile.csv"), index=False); print("written")
