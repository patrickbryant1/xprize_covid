#!/usr/bin/env bash
ADJUSTED_DATA=../data/adjusted_data.csv
OUTDIR=../results/simple_lr/

./simple_lr.py --adjusted_data $ADJUSTED_DATA --outdir $OUTDIR
