#!/usr/bin/env bash
INDIR=/home/patrick/results/COVID19/xprize/simple_rf/comparing_median/2_weeks/high/
OUTDIR=/home/patrick/results/COVID19/xprize/simple_rf/comparing_median/2_weeks/high/
./analyze.py --indir $INDIR --outdir $OUTDIR

INDIR=/home/patrick/results/COVID19/xprize/simple_rf/comparing_median/2_weeks/low/
OUTDIR=/home/patrick/results/COVID19/xprize/simple_rf/comparing_median/2_weeks/low/
./analyze.py --indir $INDIR --outdir $OUTDIR
