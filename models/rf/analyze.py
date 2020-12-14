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



def evaluate_model(feature_importances,outdir):
    '''
    Evaluate the fit model
    '''

    #Plot coefficients
    #days pred,days behind - this goes from -21 to 1,features
    feature_names = np.array(['C1_School closing', 'C2_Workplace closing', 'C3_Cancel public events', 'C4_Restrictions on gatherings', 'C5_Close public transport', 'C6_Stay at home requirements',
    'C7_Restrictions on internal movement', 'C8_International travel controls', 'H1_Public information campaigns', 'H2_Testing policy', 'H3_Contact tracing', 'H6_Facial Coverings',
    'smoothed_cases', 'cumulative_smoothed_cases', 'monthly_temperature',
    'retail_and_recreation', 'grocery_and_pharmacy', 'parks','transit_stations', 'workplaces', 'residential',
    'death_to_case_scale','case_death_delay','gross_net_income','population_density',
    #'Change in last 21 days',
    'pdi', 'idv', 'mas', 'uai', 'ltowvs', 'ivr',
    'Urban population (% of total population)','Population ages 65 and above (% of total population)',
    'GDP per capita (current US$)', 'Obesity Rate (%)', 'Cancer Rate (%)', 'Share of Deaths from Smoking (%)', 'Pneumonia Death Rate (per 100K)',
    'Share of Deaths from Air Pollution (%)','CO2 emissions (metric tons per capita)', 'Air transport (# carrier departures worldwide)', 'population'])


    feature_importances_av = np.average(feature_importances,axis=0)
    feature_importances_std = np.std(feature_importances,axis=0)
    #Normalize features
    order = np.argsort(feature_importances_av)

    fig,ax = plt.subplots(figsize=(9,4.5))
    plt.bar(np.arange(len(feature_names)),feature_importances_av[order],yerr=feature_importances_std[order])
    #plt.errorbar(np.arange(len(feature_names)),feature_importances_av,)
    #plt.yscale('log')
    plt.xticks(np.arange(len(feature_names)),labels=feature_names[order], rotation='vertical')
    plt.tight_layout()
    plt.ylabel('MDI')
    plt.title('Feature Importance (MDI)')
    plt.tight_layout()
    plt.savefig(outdir+'feature_importances_.png',format='png')
    plt.close()



#####MAIN#####
#Set font size
matplotlib.rcParams.update({'font.size': 7})
args = parser.parse_args()
indir = args.indir[0]
outdir = args.outdir[0]

#Evaluate model
feature_importances = np.load(outdir+'feature_importances.npy',allow_pickle=True)
evaluate_model(feature_importances,outdir)
