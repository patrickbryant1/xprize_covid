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
from sklearn.linear_model import TheilSenRegressor
import numpy as np


import pdb
#Arguments for argparse module:
parser = argparse.ArgumentParser(description = '''Analyze Theil-Sen model.''')

parser.add_argument('--indir', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to processed data file.')
parser.add_argument('--outdir', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to output directory. Include /in end')

def get_model_output(indir):
    '''Get model output
    '''

    #If the model has already been fitted
    corrs = []
    errors = []
    coefs = []
    intercepts = []
    for day in range(1,22):
        corrs.append(np.load(outdir+'correlations'+str(day)+'.npy',allow_pickle=True))
        errors.append(np.load(outdir+'average_error'+str(day)+'.npy',allow_pickle=True))
        coefs.append(np.load(outdir+'coefficients'+str(day)+'.npy',allow_pickle=True))
        intercepts.append(np.load(outdir+'intercepts'+str(day)+'.npy',allow_pickle=True))
    return np.array(corrs), np.array(errors), np.array(coefs), np.array(intercepts)


def predict(X_train,y_train,coefs,intercept,day):
    '''Predict to analyze error
    '''

    train_pred = np.dot(X_train,coefs)+intercept
    #Negative not allowed
    train_pred[train_pred<0]=0
    train_errors = np.absolute(train_pred-y_train)
    av_train_error = np.average(train_errors)
    std_train_error = np.std(train_errors)
    R,p = pearsonr(train_pred,y_train)
    fig,ax = plt.subplots(figsize=(6/2.54,6/2.54))
    plt.scatter(train_pred,y_train,s=1)
    plt.title('Av.er:'+str(np.round(av_train_error,3))+'|std:'+str(np.round(std_train_error,3))+'|PCC:'+str(np.round(R,2)))
    fig.tight_layout()
    plt.savefig(outdir+'train_true'+str(day)+'.png',format='png')
    plt.close()
    return av_train_error, std_train_error, R


def evaluate_model(corrs, errors, coefs, intercepts,outdir):
    '''
    Evaluate the fit model
    '''

    #Evaluate model
    results_file = open(outdir+'results.txt','w')
    #Calculate error
    results_file.write('Total 2week mae: '+str(np.sum(np.average(errors,axis=1)[:14]))+'\n')
    results_file.write('Total mae per 100000: '+str(np.sum(np.average(errors,axis=1)))+'\n')
    results_file.write('Average mae per 100000: '+str(np.average(np.average(errors,axis=1)))+'\n')
    results_file.write('Average std in mae per 100000: '+str(np.average(np.std(errors,axis=1)))+'\n')
    results_file.write('Average PCC: '+str(np.average(np.average(corrs,axis=1)))+'\n')
    results_file.write('Average std PCC: '+str(np.average(np.std(corrs,axis=1)))+'\n')
    #Plot average error per day with std
    plt.plot(range(1,errors.shape[0]+1),np.average(errors,axis=1),color='b')
    plt.fill_between(range(1,errors.shape[0]+1),np.average(errors,axis=1)-np.std(errors,axis=1),np.average(errors,axis=1)+np.std(errors,axis=1),color='b',alpha=0.5)
    plt.title('Average error with std')
    plt.xlabel('Days in the future')
    plt.ylabel('Error per 100000')
    plt.savefig(outdir+'zen_av_error.png',format='png')
    plt.close()

    #Plot correlation
    plt.plot(range(1,corrs.shape[0]+1),np.average(corrs,axis=1),color='b')
    plt.fill_between(range(1,corrs.shape[0]+1),np.average(corrs,axis=1)-np.std(corrs,axis=1),np.average(corrs,axis=1)+np.std(corrs,axis=1),color='b',alpha=0.5)
    plt.title('Pearson R')
    plt.xlabel('Days in the future')
    plt.ylabel('PCC')
    plt.savefig(outdir+'PCC.png',format='png')
    plt.close()


#####MAIN#####
#Set font size
matplotlib.rcParams.update({'font.size': 7})
args = parser.parse_args()
indir = args.indir[0]
outdir = args.outdir[0]

corrs, errors, coefs, intercepts = get_model_output(indir)

#Evaluate fit
evaluate_model(corrs, errors, coefs, intercepts,outdir)
