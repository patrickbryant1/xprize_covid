#!/usr/bin/env bash
INDIR=/home/patrick/results/COVID19/xprize/simple_rf/6_weeks_ahead/high/
OUTDIR=/home/patrick/results/COVID19/xprize/simple_rf/6_weeks_ahead/high/
./analyze.py --indir $INDIR --outdir $OUTDIR

INDIR=/home/patrick/results/COVID19/xprize/simple_rf/6_weeks_ahead/low/
OUTDIR=/home/patrick/results/COVID19/xprize/simple_rf/6_weeks_ahead/low/
./analyze.py --indir $INDIR --outdir $OUTDIR
