#!/usr/bin/env bash
ADJUSTED_DATA=../../data/adjusted_data.csv
DAYS_AHEAD=1
START_DATE='2020-01-01' #Date to start from
TRAIN_DAYS=21 #Number of days to include as the training period
FORECAST_DAYS=21
OUTDIR=/home/patrick/results/COVID19/xprize/LSTM/
./lstm.py --adjusted_data $ADJUSTED_DATA --days_ahead $DAYS_AHEAD --start_date $START_DATE --train_days $TRAIN_DAYS --forecast_days $FORECAST_DAYS --outdir $OUTDIR
