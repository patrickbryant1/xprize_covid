#!/usr/bin/env bash

START_DATE='2020-12-01' #Date to start from
END_DATE='2021-01-01' #Date to end
HISTIP=../../data/historical_ip.csv
IPCOSTS=../../data/uniform_costs.csv
OUTFILE=prescr_20201201_20210101.csv
time python3 prescribe.py -s $START_DATE -e $END_DATE -ip $HISTIP -c $IPCOSTS -o $OUTFILE
