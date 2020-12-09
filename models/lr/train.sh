#!/usr/bin/env bash
ADJUSTED_DATA=../../data/adjusted_data.csv
START_DATE='2020-01-01' #Date to start from
TRAIN_DAYS=21 #Number of days to include as the training period
FORECAST_DAYS=21
WORLD_AREA=1 #1="Europe & Central Asia"
OUTDIR=/home/patrick/results/COVID19/xprize/simple_lr/median_data/log/
./simple_lr.py --adjusted_data $ADJUSTED_DATA --start_date $START_DATE --train_days $TRAIN_DAYS --forecast_days $FORECAST_DAYS --world_area $WORLD_AREA --outdir $OUTDIR
