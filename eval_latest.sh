#!/bin/bash
# Snapshot latest.pth, evaluate it on val (GPU 1) and test (GPU 2) in parallel, append to history CSVs.
# usage: ./eval_latest.sh [run_dir]   (default: the current run)
set -u
RUN=${1:-runs/paraformer_adamw_lr1e-4_w128_bs8}
BACKBONE=${2:-vit_b16}
PY=/opt/miniconda3/envs/rishabh_sat/bin/python
EP=$(( $(grep -c "Epoch :" $RUN/log.txt) - 1 ))          # last completed epoch
E=$RUN/eval; mkdir -p $E
CK=$E/snapshot_after_epoch$EP.pth; cp $RUN/latest.pth $CK
echo "$(date '+%F %T') evaluating weights after epoch $EP" | tee -a $E/eval_history.log
$PY evaluate.py --model_path $CK --list_dir dataset/CSV_list/Chesapeake_all_val.csv  --out_dir $E/val_epoch$EP  --backbone $BACKBONE --gpu 1 > $E/val_epoch$EP.log 2>&1 &
$PY evaluate.py --model_path $CK --list_dir dataset/CSV_list/Chesapeake_all_test.csv --out_dir $E/test_epoch$EP --backbone $BACKBONE --gpu 2 > $E/test_epoch$EP.log 2>&1 &
wait
$PY append_history.py $E/val_epoch$EP/results.json  $EP $E/history_val.csv  | tee -a $E/eval_history.log
$PY append_history.py $E/test_epoch$EP/results.json $EP $E/history_test.csv | tee -a $E/eval_history.log
rm -f $CK                                                 # snapshot no longer needed; epoch_N.pth files are kept by training
echo "$(date '+%F %T') done epoch $EP: val avg=$(grep -o 'Average over states: [0-9.]*' $E/val_epoch$EP.log | cut -d' ' -f4)  test avg=$(grep -o 'Average over states: [0-9.]*' $E/test_epoch$EP.log | cut -d' ' -f4)" | tee -a $E/eval_history.log
