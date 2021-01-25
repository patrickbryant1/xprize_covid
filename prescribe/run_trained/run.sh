#!/usr/bin/env bash

START_DATE='2020-06-01' #Date to start from
END_DATE='2020-07-01' #Date to start from
HISTIP=../../data/historical_ip.csv
IPCOSTS=../../data/uniform_costs.csv
OUTFILE=prescr_20200601_20200701.csv
python prescribe.py -s $START_DATE -e $END_DATE -ip $HISTIP -w $IPCOSTS -o $OUTFILE
