#!/usr/bin/env bash
ADJUSTED_DATA=../../data/adjusted_data.csv
START_DATE='2020-01-01' #Date to start from
TRAIN_DAYS=21 #Number of days to include as the training period
FORECAST_DAYS=21
PARAMS=./params/16_0.01.params
DATADIR=/home/patrick/results/COVID19/xprize/attention/
OUTDIR=/home/patrick/results/COVID19/xprize/attention/
#/opt/singularity3/bin/singularity run --nv /home/patrick/singularity_images/tf13.sif
#Only attention, then dense and pred
#./base_attention.py --adjusted_data $ADJUSTED_DATA --start_date $START_DATE --train_days $TRAIN_DAYS --forecast_days $FORECAST_DAYS --param_combo $PARAMS --datadir $DATADIR --outdir $OUTDIR


#Dense, then attention
DATADIR=/home/patrick/results/COVID19/xprize/attention/dense/
OUTDIR=/home/patrick/results/COVID19/xprize/attention/dense/
./dense_attention.py --adjusted_data $ADJUSTED_DATA --start_date $START_DATE --train_days $TRAIN_DAYS --forecast_days $FORECAST_DAYS --param_combo $PARAMS --datadir $DATADIR --outdir $OUTDIR
