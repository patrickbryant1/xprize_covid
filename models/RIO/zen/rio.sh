#!/usr/bin/env bash
ADJUSTED_DATA=../../data/adjusted_data.csv
INDIR=/home/patrick/results/COVID19/xprize/theil_sen/
OUTDIR=/home/patrick/results/COVID19/xprize/theil_sen/residuals/
./get_residuals.py --indir $INDIR --outdir $OUTDIR
