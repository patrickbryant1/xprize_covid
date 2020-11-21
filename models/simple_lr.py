#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import sys
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

import pdb
#Arguments for argparse module:
parser = argparse.ArgumentParser(description = '''Simple linear regression model.''')

parser.add_argument('--adjusted_data', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to processed data file.')

parser.add_argument('--outdir', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to output directory. Include /in end')


def get_features(adjusted_data):
    '''Get the selected features
    '''

    selected_features = ['C1_School closing',
                        'C2_Workplace closing',
                        'C3_Cancel public events',
                        'C4_Restrictions on gatherings',
                        'C5_Close public transport',
                        'C6_Stay at home requirements',
                        'C7_Restrictions on internal movement',
                        'C8_International travel controls',
                        'H1_Public information campaigns',
                        'H2_Testing policy',
                        'H3_Contact tracing',
                        'H6_Facial Coverings', #These first 12 are the ones the prescriptor will assign
                        'Country_index',
                        'Region_index',
                        'rescaled_cases',
                        'cumulative_rescaled_cases',
                        'death_to_case_scale',
                        'case_death_delay']

    sel = adjusted_data[selected_features]

    return sel

def split_for_training(sel):
    '''Split the data for training and testing
    '''
    X = [] #Inputs
    y = [] #Targets
    countries = sel['Country_index'].unique()
    for ci in countries:
        country_data = sel[sel['Country_index']==ci]
        #Check regions
        country_regions = country_data['Region_index'].unique()
        for ri in country_regions:
            country_region_data = country_data[country_data['Region_index']==ri]
            country_region_data = country_region_data[country_region_data['cumulative_rescaled_cases']>0]
            country_region_data = country_region_data.reset_index()
            country_region_data = country_region_data.drop(columns={'index'})
            #Loop through and get the first 21 days of data
            num_intervals = int(len(country_region_data)/21)
            si = len(country_region_data)-(num_intervals*21)
            for di in range(si,len(country_region_data)-40,21):
                X.append(np.array(country_region_data.loc[di:di+20]))
                y.append(np.array(country_region_data.loc[di+21:di+21+20]['rescaled_cases']))

    return np.array(X), np.array(y)


#####MAIN#####
#Set font size
matplotlib.rcParams.update({'font.size': 7})
args = parser.parse_args()
adjusted_data = pd.read_csv(args.adjusted_data[0],
                 parse_dates=['Date'],
                 encoding="ISO-8859-1",
                 dtype={"RegionName": str,
                        "RegionCode": str,
                        "Country_index":int,
                        "Region_index":int},
                 error_bad_lines=False)
adjusted_data = adjusted_data.fillna(0)
outdir = args.outdir[0]


#Get features
try:
    X = np.load(outdir+'X.npy', allow_pickle=True)
    y = np.load(outdir+'y.npy', allow_pickle=True)
except:
    sel=get_features(adjusted_data)
    X,y = split_for_training(sel)
    #Save
    np.save(outdir+'X.npy',X)
    np.save(outdir+'y.npy',y)

X = X.reshape(2484,21*18)
for i in range(y.shape[1]):
    reg = LinearRegression().fit(X, y[:,i])
    print(i,'score',reg.score(X, y[:,i]))
    pred = reg.predict(X)
    print('Error',np.average(np.absolute(pred-y[:,i])))

'''
0 score 0.9965171788217608
Error 147.78195791300166
1 score 0.9949542110641254
Error 206.08417628527968
2 score 0.9897367636403386
Error 276.7196593242943
3 score 0.9873922778339044
Error 339.158323434575
4 score 0.9847872123800475
Error 394.95115831275984
5 score 0.981069738265629
Error 469.7773836967885
6 score 0.9744316083264396
Error 557.6283516452773
7 score 0.947937381427988
Error 774.5702332042024
8 score 0.9441666150737268
Error 813.4767450540144
9 score 0.942670262333997
Error 871.8416423164643
10 score 0.9344825469470066
Error 906.2068726931925
11 score 0.9292333514259635
Error 948.3720027828836
12 score 0.9177213373217287
Error 1011.4914917987213
13 score 0.9096159991180524
Error 1059.8138823386162
14 score 0.8996099700099277
Error 1116.0641028345824
15 score 0.885236913538402
Error 1185.2559903209649
16 score 0.8769550184409123
Error 1203.1393877270386
17 score 0.8635154292387932
Error 1237.9869729096108
18 score 0.8465172716323872
Error 1290.7524965051505
19 score 0.8364763832557539
Error 1328.2139686056391
20 score 0.8256932714137826
Error 1343.3106562059213

'''
pdb.set_trace()
