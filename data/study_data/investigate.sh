#!/usr/bin/env bash
ADJUSTED_DATA=../adjusted_data.csv
START_DATE='2020-07-01' #Date to start from
OUTDIR=/home/patrick/results/COVID19/xprize/data_relationships/
./study.py --adjusted_data $ADJUSTED_DATA --start_date $START_DATE --outdir $OUTDIR
