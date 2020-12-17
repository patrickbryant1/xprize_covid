#!/usr/bin/env bash
INDIR=/home/patrick/results/COVID19/xprize/simple_rf/comparing_median/all_regions/3_weeks/non_log/august_on/high/
OUTDIR=/home/patrick/results/COVID19/xprize/simple_rf/comparing_median/all_regions/3_weeks/non_log/august_on/high/
./analyze.py --indir $INDIR --outdir $OUTDIR

INDIR=/home/patrick/results/COVID19/xprize/simple_rf/comparing_median/all_regions/3_weeks/non_log/august_on/low/
OUTDIR=/home/patrick/results/COVID19/xprize/simple_rf/comparing_median/all_regions/3_weeks/non_log/august_on/low/
./analyze.py --indir $INDIR --outdir $OUTDIR
