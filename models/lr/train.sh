#!/usr/bin/env bash
ADJUSTED_DATA=../../data/adjusted_data.csv
START_DATE='2020-06-01' #Date to start from
OUTDIR=/home/patrick/results/COVID19/xprize/simple_lr/
./simple_lr.py --adjusted_data $ADJUSTED_DATA --start_date $START_DATE --outdir $OUTDIR
