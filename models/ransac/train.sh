#!/usr/bin/env bash
ADJUSTED_DATA=../../data/adjusted_data.csv
OUTDIR=/home/patrick/results/COVID19/xprize/ransac/
./ransac.py --adjusted_data $ADJUSTED_DATA --outdir $OUTDIR
