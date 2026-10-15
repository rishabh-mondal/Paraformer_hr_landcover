#!/bin/bash
# Waits (up to 72 h) for access to the gated facebook/pixio-vitl16 repo, then runs the Pixio-L smoke test:
#   1) download + strict weight-load check, 2) 1 training epoch on the 4-tile smoke set, 3) score 1 test tile.
# Survives Claude Code restarts (run under nohup). Progress and verdict: runs/smoke_pixio/smoke.log
cd /home/rishabh.mondal/walkblity-project/Paraformer_hr_landcover
PY=/opt/miniconda3/envs/rishabh_sat/bin/python; OUT=runs/smoke_pixio; LOG=$OUT/smoke.log; mkdir -p $OUT
log(){ echo "$(date '+%F %T') $*" >> $LOG; }
log "waiting for access to facebook/pixio-vitl16 (checks every 5 min)"
for i in $(seq 1 864); do
  $PY -c "from huggingface_hub import hf_hub_download; hf_hub_download('facebook/pixio-vitl16','config.json')" > /dev/null 2>&1 && break
  [ $i -eq 864 ] && { log "GAVE UP: no access after 72 h"; exit 1; }
  sleep 300
done
log "access granted - step 1: download + weight-load check"
$PY - >> $LOG 2>&1 <<'PY'
from huggingface_hub import snapshot_download
from transformers import PixioModel
import json, os
p = snapshot_download("facebook/pixio-vitl16", allow_patterns=["*.json", "*.safetensors", "*.md"])
c = json.load(open(os.path.join(p, "config.json")))
print("config:", {k: c.get(k) for k in ("hidden_size", "num_hidden_layers", "num_attention_heads", "patch_size", "image_size", "n_cls_tokens")})
pp = os.path.join(p, "preprocessor_config.json")
if os.path.exists(pp):
    d = json.load(open(pp)); print("preprocessor:", {k: d.get(k) for k in ("image_mean", "image_std", "size", "resample")})
m, info = PixioModel.from_pretrained("facebook/pixio-vitl16", output_loading_info=True)
print("loading info:", {k: len(v) for k, v in info.items()}, "| examples:", {k: list(v)[:3] for k, v in info.items() if v})
assert not info["missing_keys"] and not info["mismatched_keys"], "weights did not load cleanly"
print("WEIGHTS OK:", sum(t.numel() for t in m.parameters()) / 1e6, "M params")
PY
[ $? -eq 0 ] || { log "FAILED at step 1 (see above)"; exit 1; }
GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1)
log "step 2: 1 training epoch on the 4-tile smoke set, GPU $GPU"
$PY train.py --dataset Chesapeake --list_dir ./dataset/CSV_list/smoke_train.csv --batch_size 8 --max_epochs 1 --CNN_width 128 \
    --optimizer adamw --base_lr 0.0001 --num_chips_per_tile 20 --backbone pixio_l --freeze_backbone --savepath $OUT --gpu $GPU > $OUT/train.log 2>&1
grep -a "trainable\|Epoch :" $OUT/train.log | tr '\r' '\n' | grep -a "trainable\|Epoch :" >> $LOG
[ -f $OUT/latest.pth ] && ! grep -aq Traceback $OUT/train.log || { log "FAILED at step 2 (see $OUT/train.log)"; exit 1; }
log "step 3: score 1 test tile with the trained checkpoint"
$PY evaluate.py --model_path $OUT/latest.pth --list_dir dataset/CSV_list/smoke_test_1tile.csv --out_dir $OUT/eval --backbone pixio_l --gpu $GPU > $OUT/eval.log 2>&1
grep -a "^\[\|Average" $OUT/eval.log >> $LOG
[ -f $OUT/eval/results.json ] || { log "FAILED at step 3 (see $OUT/eval.log)"; exit 1; }
rm -f $OUT/*.pth                                           # 1.4 GB each; a smoke checkpoint is not worth keeping
log "SMOKE TEST PASSED (checkpoints removed; logs kept)"
