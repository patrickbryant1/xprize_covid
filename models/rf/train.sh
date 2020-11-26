#!/usr/bin/env bash

#RandomForestRegressor
ADJUSTED_DATA=../../data/adjusted_data.csv
OUTDIR=/home/patrick/results/COVID19/xprize/simple_rf/
./rf.py --adjusted_data $ADJUSTED_DATA --outdir $OUTDIR
