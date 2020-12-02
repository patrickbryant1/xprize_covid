#!/usr/bin/env bash

#Analyze
INDIR=/home/patrick/results/COVID19/xprize/theil_sen/per100000/
OUTDIR=/home/patrick/results/COVID19/xprize/theil_sen/per100000/
./analyze_parallel.py --indir $INDIR --outdir $OUTDIR
