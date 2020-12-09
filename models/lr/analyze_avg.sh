#!/usr/bin/env bash
INDIR=/home/patrick/results/COVID19/xprize/simple_lr/median_data/comparing_average/high/
OUTDIR=/home/patrick/results/COVID19/xprize/simple_lr/median_data/comparing_average/high/
./analyze_avg.py --indir $INDIR --outdir $OUTDIR

INDIR=/home/patrick/results/COVID19/xprize/simple_lr/median_data/comparing_average/low/
OUTDIR=/home/patrick/results/COVID19/xprize/simple_lr/median_data/comparing_average/low/
./analyze_avg.py --indir $INDIR --outdir $OUTDIR
