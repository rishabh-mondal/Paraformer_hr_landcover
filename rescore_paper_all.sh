#!/bin/bash
# Re-score every surviving checkpoint with the paper's Table S1 mapping (evaluate.py default), balanced over 4 GPUs.
# A job whose evaluate.py is already running (e.g. after the driver was restarted) is waited for instead of relaunched.
set -u
PY=/opt/miniconda3/envs/rishabh_sat/bin/python
cd /home/rishabh.mondal/walkblity-project/Paraformer_hr_landcover
B=paraformer_adamw_lr1e-4_w128_bs8; SAT=paraformer_dinov3_sat_frozen_adamw_lr1e-4_w128_bs8; WEB=paraformer_dinov3_web_frozen_adamw_lr1e-4_w128_bs8
LOG=runs/rescore_paper.log
job () {   # run backbone gpu epoch split [extra args]
  R=runs/$1; O=$R/eval/$5_epoch$4; L=$R/eval/$5_epoch$4.log; mkdir -p $O
  if pgrep -f -- "--out_dir $O " > /dev/null; then
    echo "$(date '+%F %T') wait  $1 $5 epoch $4 (already running)" >> $LOG
    while pgrep -f -- "--out_dir $O " > /dev/null; do sleep 60; done
  else
    echo "$(date '+%F %T') start $1 $5 epoch $4 on GPU $3" >> $LOG
    $PY evaluate.py --model_path $R/epoch_$4.pth --list_dir dataset/CSV_list/Chesapeake_all_$5.csv --out_dir $O \
        --backbone $2 --gpu $3 --mapping paper ${6:-} > $L 2>&1
  fi
  if [ -f $O/results.json ] && grep -q '"mapping": "paper"' $O/results.json; then
    $PY append_history.py $O/results.json $4 $R/eval/history_$5.csv >> $LOG 2>&1
    echo "$(date '+%F %T') done  $1 $5 epoch $4: $(grep -o 'Average over states: [0-9.]*' $L)" >> $LOG
  else
    echo "$(date '+%F %T') FAILED $1 $5 epoch $4 (see $L)" >> $LOG
  fi
}
( job $B   vit_b16    0 50 test "--save_png 4"; job $B vit_b16 0 10 test ) &
( job $SAT dinov3_sat 1 50 test "--save_png 4"; job $B vit_b16 1 0 val; job $B vit_b16 1 10 val ) &
( job $WEB dinov3_web 2 50 test "--save_png 4" ) &
( job $B   vit_b16    3 0 test; job $WEB dinov3_web 3 10 test ) &
wait
echo "$(date '+%F %T') ALL DONE" >> $LOG
