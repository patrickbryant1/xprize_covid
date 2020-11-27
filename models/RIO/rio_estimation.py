#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import sys
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


from scipy.stats import pearsonr
import numpy as np


import pdb
#Arguments for argparse module:
parser = argparse.ArgumentParser(description = '''Calculate the parameters for RIO.''')

parser.add_argument('--pred_train', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to predictions of train data.')
parser.add_argument('--true_train', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to true values of train data.')
parser.add_argument('--outdir', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to output directory. Include /in end')


#####FUNCTIONS#####
'''
1. Need to estimate the paramters l and sigma for a Radial Basis Function (RBF) kernel
that learns the redisuals (true-pred) by minimizing the log marginal likelihood: log p(r|X,y_pred),
where X is the inputs used to train the neural net and y_pred its resulting predictions.

2. After the parameters l and sigma have been learned, the predicted residuals can be used
to calibrate the point predictions of the NN as well as providing the variance for the predictions.
y_adj = (y_pred+r)+-r_var 
'''


#####MAIN#####
#Set font size
matplotlib.rcParams.update({'font.size': 7})
args = parser.parse_args()

pred_train = np.load(args.pred_train[0],allow_pickle=True)
true_train = np.load(args.true_train[0],allow_pickle=True)
outdir = args.outdir[0]

#Get features
X_train,y_train,X_test,y_test,populations,regions  = get_features(indir)

#Get residuals for RIO
get_residuals(X_train,y_train,indir,outdir)
