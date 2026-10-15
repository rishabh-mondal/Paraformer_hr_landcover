#!/bin/bash
# Score every saved checkpoint (epochs 0,10,...,90 and the final 99) of the three runs on the test set,
# ONE AT A TIME ON GPU 0 ONLY, in epoch order (epoch 0: baseline, sat, web; then epoch 10; ...).
# - skips checkpoints already scored with the paper's Table S1 mapping (history_test.csv, mapping == paper)
# - waits for checkpoints that training has not saved yet (log.txt "save model to .../epoch_E.pth")
# - after each result: updates runs/test_results_tableS1.csv and rebuilds ../slides_chesapeake.pdf in the background
# Restart-safe and session-independent: run under nohup. Log: runs/eval_all_epochs.log
cd /home/rishabh.mondal/walkblity-project/Paraformer_hr_landcover
PY=/opt/miniconda3/envs/rishabh_sat/bin/python; LOG=runs/eval_all_epochs.log; GPU=0
RUNS="paraformer_adamw_lr1e-4_w128_bs8:vit_b16 paraformer_dinov3_sat_frozen_adamw_lr1e-4_w128_bs8:dinov3_sat paraformer_dinov3_web_frozen_adamw_lr1e-4_w128_bs8:dinov3_web"
EPOCHS="0 10 20 30 40 50 60 70 80 90 99"
log(){ echo "$(date '+%F %T') $*" >> $LOG; }
scored(){ $PY -c "import pandas as pd,sys; h=pd.read_csv('runs/$1/eval/history_test.csv'); sys.exit(0 if ('mapping' in h and ((h.mapping=='paper')&(h.epoch==$2)).any()) else 1)" 2>/dev/null; }
rebuild(){ ( flock -n 9 || exit 0
             $PY collect_results.py > /dev/null 2>&1
             cd .. && $PY make_slides.py > /dev/null 2>&1 && echo "$(date '+%F %T') slides rebuilt" >> Paraformer_hr_landcover/$LOG
           ) 9> runs/.slides.lock & }
log "=== queue started: epochs [$EPOCHS] x 3 runs on GPU $GPU"
for E in $EPOCHS; do
  for RB in $RUNS; do
    R=${RB%%:*}; BB=${RB##*:}; NAME=$(echo $R | sed 's/paraformer_//; s/_adamw_lr1e-4_w128_bs8//; s/_frozen//; s/adamw_lr1e-4_w128_bs8/baseline/')
    [ -z "$NAME" ] && NAME=baseline
    if scored $R $E; then log "skip  $NAME epoch $E (already scored, Table S1)"; continue; fi
    CK=runs/$R/epoch_$E.pth; W=0
    until [ -f $CK ] && grep -q "save model to runs/$R/epoch_$E.pth" runs/$R/log.txt; do
      [ $W -eq 0 ] && log "wait  $NAME epoch $E (checkpoint not saved yet)"; W=1; sleep 300
    done
    O=runs/$R/eval/test_epoch$E; L=runs/$R/eval/test_epoch$E.log; mkdir -p $O
    log "start $NAME epoch $E on GPU $GPU"
    $PY evaluate.py --model_path $CK --list_dir dataset/CSV_list/Chesapeake_all_test.csv --out_dir $O \
        --backbone $BB --gpu $GPU --mapping paper > $L 2>&1
    if [ -f $O/results.json ] && grep -q '"mapping": "paper"' $O/results.json; then
      $PY append_history.py $O/results.json $E runs/$R/eval/history_test.csv > /dev/null 2>&1
      log "done  $NAME epoch $E: $(grep -o 'Average over states: [0-9.]*' $L) | $(grep -o 'pooled: [0-9.]*' $L)"
      rebuild
    else
      log "FAILED $NAME epoch $E (see $L)"
    fi
  done
done
wait
( flock 9; $PY collect_results.py > /dev/null 2>&1; cd .. && $PY make_slides.py > /dev/null 2>&1 ) 9> runs/.slides.lock
log "=== ALL DONE: runs/test_results_tableS1.csv and slides_chesapeake.pdf are final"
