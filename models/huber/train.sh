#!/usr/bin/env bash
ADJUSTED_DATA=../../data/adjusted_data.csv
START_DATE='2020-01-01' #Date to start from
TRAIN_DAYS=28 #Number of days to include as the training period
OUTDIR=/home/patrick/results/COVID19/xprize/huber/
./huber.py --adjusted_data $ADJUSTED_DATA  --start_date $START_DATE --train_days $TRAIN_DAYS --outdir $OUTDIR
