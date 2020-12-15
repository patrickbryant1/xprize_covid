#!/usr/bin/env bash
ADJUSTED_DATA=../../data/adjusted_data.csv
START_DATE='2020-06-01' #Date to start from
TRAIN_DAYS=21 #Number of days to include as the training period
FORECAST_DAYS=21
THRESHOLD=1.8
OUTDIR=/home/patrick/results/COVID19/xprize/iterative/
./iter_net.py --adjusted_data $ADJUSTED_DATA --start_date $START_DATE --train_days $TRAIN_DAYS --forecast_days $FORECAST_DAYS --threshold $THRESHOLD --outdir $OUTDIR
