#!/usr/bin/env bash

TRAIN_PREDS=/home/patrick/results/COVID19/xprize/theil_sen/per100000/residuals/train_preds.npy
RESIDUALS=/home/patrick/results/COVID19/xprize/theil_sen/per100000/residuals/residuals.npy
OUTDIR=/home/patrick/results/COVID19/xprize/theil_sen/per100000/residuals/MQR/
./auto.py --train_preds $TRAIN_PREDS --residuals $RESIDUALS --outdir $OUTDIR
