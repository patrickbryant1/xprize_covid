#!/usr/bin/env bash
ADJUSTED_DATA=../../data/adjusted_data.csv
TEMP_DATA=../../data/temp_1991_2016.csv
IPCOSTS=../../data/uniform_costs.csv
START_DATE='2020-06-01' #Date to start from
TRAIN_DAYS=21 #Number of days to include as the training period
FORECAST_DAYS=21
THRESHOLD=1.8
OUTDIR=/home/patrick/results/COVID19/xprize/prescriptor/
./deap_nsga3.py --adjusted_data $ADJUSTED_DATA --temp_data $TEMP_DATA --ip_costs $IPCOSTS --start_date $START_DATE --train_days $TRAIN_DAYS --forecast_days $FORECAST_DAYS --threshold $THRESHOLD --outdir $OUTDIR
