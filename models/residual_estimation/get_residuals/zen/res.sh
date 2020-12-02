#!/usr/bin/env bash
ADJUSTED_DATA=../../data/adjusted_data.csv
INDIR=/home/patrick/results/COVID19/xprize/theil_sen/per100000/
OUTDIR=/home/patrick/results/COVID19/xprize/theil_sen/per100000/residuals/
./get_residuals.py --indir $INDIR --outdir $OUTDIR
