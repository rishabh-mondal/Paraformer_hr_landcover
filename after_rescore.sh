#!/bin/bash
# Survives Claude Code session restarts (run under nohup): when rescore_paper_all.sh logs ALL DONE,
# rebuild the epoch-50 comparison images and the slide deck with the Table S1 results.
cd /home/rishabh.mondal/walkblity-project/Paraformer_hr_landcover
PY=/opt/miniconda3/envs/rishabh_sat/bin/python; LOG=runs/rescore_paper.log
until grep -q "ALL DONE" $LOG 2>/dev/null; do sleep 120; done
$PY make_test_figures.py 50 >> $LOG 2>&1 && echo "$(date '+%F %T') figures rebuilt (runs/compare_epoch50)" >> $LOG
cd .. && $PY make_slides.py >> Paraformer_hr_landcover/$LOG 2>&1 && echo "$(date '+%F %T') slides rebuilt (slides_chesapeake.pdf)" >> Paraformer_hr_landcover/$LOG
echo "$(date '+%F %T') FINISHED" >> Paraformer_hr_landcover/$LOG
