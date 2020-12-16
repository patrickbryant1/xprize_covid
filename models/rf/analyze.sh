#!/usr/bin/env bash
INDIR=/home/patrick/results/COVID19/xprize/simple_rf/comparing_median/all_regions/2_weeks/non_log/high/
OUTDIR=/home/patrick/results/COVID19/xprize/simple_rf/comparing_median/all_regions/2_weeks/non_log/high/
./analyze.py --indir $INDIR --outdir $OUTDIR

INDIR=/home/patrick/results/COVID19/xprize/simple_rf/comparing_median/all_regions/2_weeks/non_log/low/
OUTDIR=/home/patrick/results/COVID19/xprize/simple_rf/comparing_median/all_regions/2_weeks/non_log/low/
./analyze.py --indir $INDIR --outdir $OUTDIR
