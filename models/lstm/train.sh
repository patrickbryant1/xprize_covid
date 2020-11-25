#!/usr/bin/env bash
ADJUSTED_DATA=../../data/adjusted_data.csv
OUTDIR=/home/patrick/results/COVID19/xprize/lstm/
./lstm_net.py --adjusted_data $ADJUSTED_DATA --outdir $OUTDIR
