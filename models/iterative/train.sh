#!/usr/bin/env bash
ADJUSTED_DATA=../../data/adjusted_data.csv
START_DATE='2020-11-01' #Date to start from
TRAIN_DAYS=21 #Number of days to include as the training period
FORECAST_DAYS=21
THRESHOLD=1.8
BASEDIR=/home/patrick/results/COVID19/xprize/simple_rf/comparing_median/subregions/3_weeks

#WORLD_AREA
#'Latin America & Caribbean':1, 'South Asia':2, 'Sub-Saharan Africa':3,
 #'Europe & Central Asia':4, 'Middle East & North Africa':5,
 #'East Asia & Pacific':6, 'North America':7
for WORLD_AREA in 7 #{1..7}
do
  OUTDIR=$BASEDIR'/wa'$WORLD_AREA
  mkdir $OUTDIR
  mkdir $OUTDIR/high
  mkdir $OUTDIR/low
  ./rf.py --adjusted_data $ADJUSTED_DATA --start_date $START_DATE --train_days $TRAIN_DAYS --forecast_days $FORECAST_DAYS --world_area $WORLD_AREA --threshold $THRESHOLD --outdir $OUTDIR$i/
done
