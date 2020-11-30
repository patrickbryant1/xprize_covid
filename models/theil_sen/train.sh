#!/usr/bin/env bash
ADJUSTED_DATA=../../data/adjusted_data.csv
OUTDIR=/home/patrick/results/COVID19/xprize/theil_sen/
#./zen.py --adjusted_data $ADJUSTED_DATA --outdir $OUTDIR

#Analyze
INDIR=/home/patrick/results/COVID19/xprize/theil_sen/
OUTDIR=/home/patrick/results/COVID19/xprize/theil_sen/
./analyze_parallel.py --indir $INDIR --outdir $OUTDIR
