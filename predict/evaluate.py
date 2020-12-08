#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--in_csv', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to input csv to be evaluated.')
parser.add_argument('--outdir', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to output directory. Include /in end')


def calc_error_pcc(pred,true):
    '''
    Evalute predictions
    '''

    #Calculate the cumulative error
    cumulative_error = np.sum(np.absolute(pred-true))

    #Evaluate PCC
    R,p = pearsonr(pred,true)

    return cumulative_error, R


#####MAIN#####
matplotlib.rcParams.update({'font.size': 7})
args = parser.parse_args()
in_csv = pd.read_csv(args.in_csv[0],
                          parse_dates=['Date'],
                          encoding="ISO-8859-1",
                          dtype={"RegionName": str},
                          error_bad_lines=True)
outdir = args.outdir[0]

#Define region
in_csv['GeoID'] = in_csv['CountryName'] + '__' + in_csv['RegionName'].astype(str)
#Evaluate all regions and save
all_cumulative_errors = []
all_PCC = []
for region in in_csv['GeoID'].unique():
    geo_data = in_csv[in_csv['GeoID']==region]
    population = geo_data['population'].values[0]
    geo_pred = np.array(geo_data['PredictedDailyNewCases'])/(population/100000)
    geo_true = np.array(geo_data['smoothed_cases'])//(population/100000)
    #Calculate error
    cumulative_error, R = calc_error_pcc(geo_pred, geo_true)
    all_cumulative_errors.append(cumulative_error)
    all_PCC.append(R)

all_cumulative_errors = np.array(all_cumulative_errors)
all_PCC = np.array(all_PCC)

print(min(in_csv['Date']),max(in_csv['Date']))
print('Average cumulative error:', np.average(all_cumulative_errors))
print('Average PCC:', np.average(all_PCC[~np.isnan(all_PCC)]))
