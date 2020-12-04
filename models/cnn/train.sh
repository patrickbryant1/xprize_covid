#!/usr/bin/env bash
ADJUSTED_DATA=../../data/adjusted_data.csv
START_DATE='2020-01-01' #Date to start from
TRAIN_DAYS=21 #Number of days to include as the training period
FORECAST_DAYS=21
PARAMS=./params/16_12_10_0.01_1.params
DATADIR=/home/patrick/results/COVID19/xprize/CNN/
OUTDIR=/home/patrick/results/COVID19/xprize/CNN/
#./cnn.py --adjusted_data $ADJUSTED_DATA --start_date $START_DATE --train_days $TRAIN_DAYS --forecast_days $FORECAST_DAYS --param_combo $PARAMS --datadir $DATADIR --outdir $OUTDIR

./cnn_serial_dil.py --adjusted_data $ADJUSTED_DATA --start_date $START_DATE --train_days $TRAIN_DAYS --forecast_days $FORECAST_DAYS --param_combo $PARAMS --datadir $DATADIR --outdir $OUTDIR
