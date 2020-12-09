#!/usr/bin/env bash
INDIR=/home/patrick/results/COVID19/xprize/simple_lr/median_data/above100/
OUTDIR=/home/patrick/results/COVID19/xprize/simple_lr/median_data/above100/
./analyze.py --indir $INDIR --outdir $OUTDIR

INDIR=/home/patrick/results/COVID19/xprize/simple_lr/median_data/low/
OUTDIR=/home/patrick/results/COVID19/xprize/simple_lr/median_data/low/
./analyze.py --indir $INDIR --outdir $OUTDIR
