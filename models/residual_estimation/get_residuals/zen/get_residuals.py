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
parser = argparse.ArgumentParser(description = '''Get the residuals for RIO.''')

parser.add_argument('--indir', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to processed data file.')

parser.add_argument('--outdir', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to output directory. Include /in end')


def get_features(indir):
    '''Get the selected features
    '''

    #Get features
    X_train = np.load(indir+'X_train.npy', allow_pickle=True)
    y_train = np.load(indir+'y_train.npy', allow_pickle=True)
    X_test = np.load(indir+'X_test.npy', allow_pickle=True)
    y_test = np.load(indir+'y_test.npy', allow_pickle=True)
    populations = np.load(indir+'populations.npy', allow_pickle=True)
    regions = np.load(indir+'regions.npy', allow_pickle=True)

    return X_train,y_train,X_test,y_test,populations,regions

def get_residuals(X_train,y_train,indir,outdir):
    '''Get the residuals for the data
    '''

    #Get the coefficients and intercepts, predict and calculate residuals
    residuals = []
    preds = []
    for day in range(1,22):
        coefs= np.load(indir+'coefficients/coefficients'+str(day)+'.npy',allow_pickle=True)
        intercept = np.load(indir+'intercepts/intercept'+str(day)+'.npy',allow_pickle=True)
        pred = np.dot(X_train,coefs)+intercept
        #Negative not allowed
        pred[pred<0]=0
        residuals.append(y_train[:,day-1]-pred)
        preds.append(pred)
        #Plot residuals
        #plt.plot(np.arange(residuals[-1].shape[0]),residuals[-1])
        #plt.savefig(outdir+'residuals'+str(day)+'.png',format='png',dpi=300)
        #plt.close()


    #Save
    np.save(outdir+'residuals.npy',np.array(residuals))
    np.save(outdir+'train_preds.npy',np.array(preds))
    print('Calculated residuals.')

    #Visualize the residual development
    residuals = np.array(residuals)
    for i in range(residuals.shape[1]):
        plt.plot(np.arange(residuals.shape[0]),residuals[:,i],linewidth=0.5,alpha=0.1)
    plt.show()
    pdb.set_trace()
#####MAIN#####
#Set font size
matplotlib.rcParams.update({'font.size': 7})
args = parser.parse_args()

indir = args.indir[0]
outdir = args.outdir[0]

#Get features
X_train,y_train,X_test,y_test,populations,regions  = get_features(indir)

#Get residuals for RIO
get_residuals(X_train,y_train,indir,outdir)
