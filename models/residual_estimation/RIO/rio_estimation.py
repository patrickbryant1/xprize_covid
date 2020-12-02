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

def get_gpr_model(X_train,pred_train,true_train,outdir):
    '''Create a GPR model in pymc3
    '''

    n=X_train.shape[0] #Number of data points
    batch_size=128
    Xy=np.append(X_train,np.array([y_train]).T,axis=1)
    Xy = Xy
    #Independent variables
    batch = pm.Minibatch(Xy,batch_size)



    # Find the parameters for the relation between X and Y.
    with pm.Model() as model:
        # Prior parameters.
        #mus = [pm.Normal(str(i), mu=0, sigma=10) for i in range(Xy.shape[1])]

        beta = pm.Normal('beta', mu=0, sigma=10,shape=X_train.shape[1])
        # c = pm.Normal('c', mu=0, sigma=10)

        sigma = pm.HalfNormal('sigma', sigma=10)

        # Relation between X and the mean of Y.
        #mu = a + b * C1_batch + c * C2_batch
        mu = pm.math.dot(batch[:,:-1],beta)
        # Observed output Y.
        y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma,
                      observed=batch[:,-1], total_size=n)

        approx = pm.fit(100000, method='advi', callbacks=[pm.callbacks.CheckParametersConvergence(tolerance=1e-4)])


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
