#!/usr/bin/env bash
ADJUSTED_DATA=../../data/adjusted_data.csv
INDIR=/home/patrick/results/COVID19/xprize/theil_sen/
OUTDIR=/home/patrick/results/COVID19/xprize/theil_sen/residuals/
#./get_residuals.py --indir $INDIR --outdir $OUTDIR

TRAIN_PRED=/home/patrick/results/COVID19/xprize/theil_sen/residuals/train_preds.npy
TRAIN_TRUE=/home/patrick/results/COVID19/xprize/theil_sen/y_train.npy
OUTDIR=/home/patrick/results/COVID19/xprize/theil_sen/residuals/
../rio_estimation.py --pred_train $TRAIN_PRED --true_train $TRAIN_TRUE --outdir $OUTDIR
