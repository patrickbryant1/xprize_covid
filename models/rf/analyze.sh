#!/usr/bin/env bash
INDIR=/home/patrick/results/COVID19/xprize/simple_rf/comparing_median/high/
OUTDIR=/home/patrick/results/COVID19/xprize/simple_rf/comparing_median/high/
./analyze.py --indir $INDIR --outdir $OUTDIR

INDIR=/home/patrick/results/COVID19/xprize/simple_rf/comparing_median/low/
OUTDIR=/home/patrick/results/COVID19/xprize/simple_rf/comparing_median/low/
./analyze.py --indir $INDIR --outdir $OUTDIR
