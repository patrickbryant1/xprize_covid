#!/usr/bin/env bash
PRED_DIR=../../standard_predictor/
IPCOSTS=../../../data/uniform_costs.csv
START_DATE='2020-06-01' #Date to start from
FORECAST_DAYS=21 #Number of days to design prescriptions for in each step
OUTDIR=/home/patrick/results/COVID19/xprize/prescriptor/standard/
./deap_nsga3.py --pred_dir $PRED_DIR --ip_costs $IPCOSTS --start_date $START_DATE --forecast_days $FORECAST_DAYS --outdir $OUTDIR
