#!/usr/bin/env bash
DAYS_AHEAD=21
XTRAIN=/home/patrick/results/COVID19/xprize/theil_sen/per100000/X_train.npy
TRAIN_PREDS=/home/patrick/results/COVID19/xprize/theil_sen/per100000/residuals/train_preds.npy
RESIDUALS=/home/patrick/results/COVID19/xprize/theil_sen/per100000/residuals/residuals.npy
OUTDIR=/home/patrick/results/COVID19/xprize/theil_sen/per100000/residuals/MQR
./mqr_net.py --days_ahead $DAYS_AHEAD --X_train $XTRAIN --train_preds $TRAIN_PREDS --residuals $RESIDUALS --outdir $OUTDIR
