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
        #if f==2:
        #    continue
        corrs.append(np.load(outdir+'corrs'+str(f)+'.npy',allow_pickle=True))
        errors.append(np.load(outdir+'errors'+str(f)+'.npy',allow_pickle=True))
        coefs.append(np.load(outdir+'coefs'+str(f)+'.npy',allow_pickle=True))
        intercepts.append(np.load(outdir+'intercepts'+str(f)+'.npy',allow_pickle=True))
    return np.array(corrs), np.array(errors), np.array(coefs), np.array(intercepts)


def evaluate_model(corrs, errors, coefs, intercepts,outdir):
    '''
    Evaluate the fit model
    '''

    #Plot coefficients
    #days pred,days behind - this goes from -21 to 1,features
    feature_names = ['C1_School closing', 'C2_Workplace closing', 'C3_Cancel public events', 'C4_Restrictions on gatherings', 'C5_Close public transport', 'C6_Stay at home requirements',
    'C7_Restrictions on internal movement', 'C8_International travel controls', 'H1_Public information campaigns', 'H2_Testing policy', 'H3_Contact tracing', 'H6_Facial Coverings',
    'smoothed_cases', 'cumulative_smoothed_cases', 'monthly_temperature', 'retail_and_recreation', 'grocery_and_pharmacy', 'parks','transit_stations', 'workplaces', 'residential',
    'death_to_case_scale','case_death_delay','gross_net_income','population_density','Change in last 21 days','pdi', 'idv', 'mas', 'uai', 'ltowvs', 'ivr','population']


    coef_av = np.average(coefs,axis=0)

    #Normalize features
    #coef_av = coef_av[0]/max(np.absolute(coef_av[0]))

    fig,ax = plt.subplots(figsize=(9,4.5))
    plt.bar(np.arange(len(feature_names)),coef_av[0])
    plt.xticks(np.arange(len(feature_names)),labels=feature_names, rotation='vertical')
    plt.tight_layout()
    plt.savefig(outdir+'coefs.png',format='png')
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
