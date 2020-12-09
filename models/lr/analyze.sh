#!/usr/bin/env bash
INDIR=/home/patrick/results/COVID19/xprize/simple_lr/median_data/log/above100/
OUTDIR=/home/patrick/results/COVID19/xprize/simple_lr/median_data/log/above100/
./analyze.py --indir $INDIR --outdir $OUTDIR

INDIR=/home/patrick/results/COVID19/xprize/simple_lr/median_data/log/low/
OUTDIR=/home/patrick/results/COVID19/xprize/simple_lr/median_data/log/low/
./analyze.py --indir $INDIR --outdir $OUTDIR
