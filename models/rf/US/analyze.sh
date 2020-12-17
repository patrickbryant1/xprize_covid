#!/usr/bin/env bash
INDIR=/home/patrick/results/COVID19/xprize/simple_rf/comparing_median/all_regions/3_weeks/non_log/US/high/
OUTDIR=/home/patrick/results/COVID19/xprize/simple_rf/comparing_median/all_regions/3_weeks/non_log/US/high/
./analyze.py --indir $INDIR --outdir $OUTDIR

INDIR=/home/patrick/results/COVID19/xprize/simple_rf/comparing_median/all_regions/3_weeks/non_log/US/low/
OUTDIR=/home/patrick/results/COVID19/xprize/simple_rf/comparing_median/all_regions/3_weeks/non_log/US/low/
./analyze.py --indir $INDIR --outdir $OUTDIR
