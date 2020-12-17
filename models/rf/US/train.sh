#!/usr/bin/env bash
ADJUSTED_DATA=../../../data/adjusted_data.csv
START_DATE='2020-06-01' #Date to start from
TRAIN_DAYS=21 #Number of days to include as the training period
FORECAST_DAYS=21
THRESHOLD=1.8
SEX_ETH_AGE_DATA=../../../data/us_sex_eth_age_per_state.csv
OUTDIR=/home/patrick/results/COVID19/xprize/simple_rf/comparing_median/all_regions/3_weeks/non_log/US/

#WORLD_AREA
#'Latin America & Caribbean':1, 'South Asia':2, 'Sub-Saharan Africa':3,
 #'Europe & Central Asia':4, 'Middle East & North Africa':5,
 #'East Asia & Pacific':6, 'North America':7
WORLD_AREA=1 #Not used currently
./rf.py --adjusted_data $ADJUSTED_DATA --start_date $START_DATE --train_days $TRAIN_DAYS --forecast_days $FORECAST_DAYS --world_area $WORLD_AREA --threshold $THRESHOLD --sex_eth_age_data $SEX_ETH_AGE_DATA --outdir $OUTDIR
