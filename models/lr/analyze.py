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
from scipy import stats
import numpy as np



import pdb
#Arguments for argparse module:
parser = argparse.ArgumentParser(description = '''Analyze model.''')

parser.add_argument('--indir', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to directory with data.')
parser.add_argument('--outdir', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to output directory. Include /in end')


def get_model_output(NFOLD,indir):
    '''Get model output
    '''

    #If the model has already been fitted
    corrs = []
    errors = []
    coefs = []
    intercepts = []
    for f in range(1,NFOLD+1):
        corrs.append(np.load(outdir+'corrs'+str(f)+'.npy',allow_pickle=True))
        errors.append(np.load(outdir+'errors'+str(f)+'.npy',allow_pickle=True))
        coefs.append(np.load(outdir+'coefs'+str(f)+'.npy',allow_pickle=True))
        intercepts.append(np.load(outdir+'intercepts'+str(f)+'.npy',allow_pickle=True))
    return np.array(corrs), np.array(errors), np.array(coefs), np.array(intercepts)


def evaluate_model(corrs, errors, coefs, intercepts,outdir):
    '''
    Evaluate the fit model
    '''

    #Evaluate model
    results_file = open(outdir+'results.txt','w')
    #Calculate error
    results_file.write('Total 2week mae: '+str(np.sum(np.average(errors,axis=0)[:14]))+'\n')
    results_file.write('Total mae per 100000: '+str(np.sum(np.average(errors,axis=0)))+'\n')
    results_file.write('Std in total mae per 100000: '+str(np.sum(np.std(errors,axis=0)))+'\n')
    results_file.write('Average mae per 100000: '+str(np.average(np.average(errors,axis=0)))+'\n')
    results_file.write('Average std in mae per 100000: '+str(np.average(np.std(errors,axis=0)))+'\n')
    results_file.write('Average PCC: '+str(np.average(np.average(corrs,axis=0)))+'\n')
    results_file.write('Average std PCC: '+str(np.average(np.std(corrs,axis=0)))+'\n')
    #Plot average error per day with std
    plt.plot(range(1,errors.shape[1]+1),np.average(errors,axis=0),color='b')
    plt.fill_between(range(1,errors.shape[1]+1),np.average(errors,axis=0)-np.std(errors,axis=0),np.average(errors,axis=0)+np.std(errors,axis=0),color='b',alpha=0.5)
    plt.title('Average error with std')
    plt.xlabel('Days in the future')
    plt.ylabel('Error per 100000')
    plt.savefig(outdir+'lr_av_error.png',format='png')
    plt.close()

    #Plot correlation
    plt.plot(range(1,corrs.shape[1]+1),np.average(corrs,axis=0),color='b')
    plt.fill_between(range(1,corrs.shape[1]+1),np.average(corrs,axis=0)-np.std(corrs,axis=0),np.average(corrs,axis=0)+np.std(corrs,axis=0),color='b',alpha=0.5)
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

#Evaluate model
corrs, errors, coefs, intercepts = get_model_output(5,indir)
evaluate_model(corrs, errors, coefs, intercepts,outdir)
