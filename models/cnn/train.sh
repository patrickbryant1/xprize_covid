#!/usr/bin/env bash
ADJUSTED_DATA=../../data/adjusted_data.csv
START_DATE='2020-01-01' #Date to start from
TRAIN_DAYS=21 #Number of days to include as the training period
FORECAST_DAYS=21
PARAMS=./params/32_3_15_0.001_1_1.params #./params/64_12_10_0.001_3_1.params
DATADIR=/home/patrick/results/COVID19/xprize/CNN/
OUTDIR=/home/patrick/results/COVID19/xprize/CNN/
/opt/singularity3/bin/singularity run --nv /home/patrick/singularity_images/tf13.sif python3 ./cnn.py --adjusted_data $ADJUSTED_DATA --start_date $START_DATE --train_days $TRAIN_DAYS --forecast_days $FORECAST_DAYS --param_combo $PARAMS --datadir $DATADIR --outdir $OUTDIR

#./cnn_serial_dil.py --adjusted_data $ADJUSTED_DATA --start_date $START_DATE --train_days $TRAIN_DAYS --forecast_days $FORECAST_DAYS --param_combo $PARAMS --datadir $DATADIR --outdir $OUTDIR
