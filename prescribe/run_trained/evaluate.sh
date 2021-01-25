#!/usr/bin/env bash
PRESCRIPTIONS=./prescr_20201201_20210101.csv
OUTDIR=./
../evaluate/evaluate.py --in_csv $PRESCRIPTIONS --outdir $OUTDIR
