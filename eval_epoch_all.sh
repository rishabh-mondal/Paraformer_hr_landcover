#!/bin/bash
# Test-set evaluation of one saved epoch for all three runs, each on its own training GPU, with small PNG predictions.
# usage: ./eval_epoch_all.sh 50
set -u
EP=$1; PY=/opt/miniconda3/envs/rishabh_sat/bin/python
cd /home/rishabh.mondal/walkblity-project/Paraformer_hr_landcover
run_one () {   # run_dir backbone gpu
  R=runs/$1; O=$R/eval/test_epoch$EP
  $PY evaluate.py --model_path $R/epoch_$EP.pth --list_dir dataset/CSV_list/Chesapeake_all_test.csv --out_dir $O \
      --backbone $2 --gpu $3 --save_png 4 > $R/eval/test_epoch$EP.log 2>&1 \
  && $PY append_history.py $O/results.json $EP $R/eval/history_test.csv >> $R/eval/eval_history.log \
  && echo "$(date '+%F %T') done test epoch $EP ($2, GPU $3, png x4)" >> $R/eval/eval_history.log
}
run_one paraformer_adamw_lr1e-4_w128_bs8 vit_b16 0 &
run_one paraformer_dinov3_sat_frozen_adamw_lr1e-4_w128_bs8 dinov3_sat 1 &
run_one paraformer_dinov3_web_frozen_adamw_lr1e-4_w128_bs8 dinov3_web 2 &
wait
echo "$(date '+%F %T') ALL DONE epoch $EP" > runs/eval_epoch${EP}_all.done
