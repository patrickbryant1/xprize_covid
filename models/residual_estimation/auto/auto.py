#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import sys
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from statsmodels.tsa.ar_model import AutoReg
from scipy.stats import pearsonr
import numpy as np


import pdb
#Arguments for argparse module:
parser = argparse.ArgumentParser(description = '''Estimate residuals with autoregression.''')

parser.add_argument('--train_preds', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to predictions of train data.')
parser.add_argument('--residuals', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to true values of train data.')
parser.add_argument('--outdir', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to output directory. Include /in end')


#####FUNCTIONS#####
def fit_model(residuals):
    '''
    Fit an autoregressive model on the assumption that the error propagates
    (if predicted above --> more above (below --> more below))
    res(t+1) = b0 + b1*res(t-1)+b2*res(t-2)+...+bn*res(t-n)
    '''
    pdb.set_trace()
    # model the training set residuals
    window = 15
    model = AutoReg(residuals, lags=15)
    model_fit = model.fit()
    coef = model_fit.params
#####MAIN#####
#Set font size
matplotlib.rcParams.update({'font.size': 7})
args = parser.parse_args()

train_preds = np.load(args.train_preds[0],allow_pickle=True)
residuals = np.load(args.residuals[0],allow_pickle=True)
outdir = args.outdir[0]

#Fit model
fit_model(residuals)
