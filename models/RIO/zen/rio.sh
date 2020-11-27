#!/usr/bin/env bash
ADJUSTED_DATA=../../data/adjusted_data.csv
INDIR=/home/patrick/results/COVID19/xprize/theil_sen/
OUTDIR=/home/patrick/results/COVID19/xprize/theil_sen/residuals/
#./get_residuals.py --indir $INDIR --outdir $OUTDIR

PRED_TRAIN=/home/patrick/results/COVID19/xprize/theil_sen/residuals/train_preds.npy
TRUE_TRAIN=/home/patrick/results/COVID19/xprize/theil_sen/y_train.npy
OUTDIR=/home/patrick/results/COVID19/xprize/theil_sen/residuals/
../rio_estimation.py --pred_train $PRED_TRAIN --true_train $TRUE_TRAIN --outdir $OUTDIR
