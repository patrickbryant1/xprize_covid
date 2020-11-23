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
                        'case_death_delay',
                        'population']

    sel = adjusted_data[selected_features]

    return sel

def split_for_training(sel):
    '''Split the data for training and testing
    '''
    X_train = [] #Inputs
    y_train = [] #Targets
    X_test = [] #Inputs
    y_test = [] #Targets
    countries = sel['Country_index'].unique()
    for ci in countries:
        country_data = sel[sel['Country_index']==ci]
        #Check regions
        country_regions = country_data['Region_index'].unique()
        for ri in country_regions:
            country_region_data = country_data[country_data['Region_index']==ri]
            country_region_data = country_region_data[country_region_data['cumulative_rescaled_cases']>0]
            country_region_data = country_region_data.reset_index()

            try:
                country_index = country_region_data.loc[0,'Country_index']
                region_index = country_region_data.loc[0,'Region_index']
                death_to_case_scale = country_region_data.loc[0,'death_to_case_scale']
                case_death_delay = country_region_data.loc[0,'case_death_delay']
                population = country_region_data.loc[0,'population']
                country_region_data = country_region_data.drop(columns={'index','Country_index', 'Region_index','death_to_case_scale', 'case_death_delay', 'population'})

                #Normalize the cases by 100'000 population
                country_region_data['rescaled_cases']=country_region_data['rescaled_cases']/population
                country_region_data['cumulative_rescaled_cases']=country_region_data['cumulative_rescaled_cases']/population
            except:
                pdb.set_trace()
            #Loop through and get the first 21 days of data
            for di in range(len(country_region_data)-41):
                xi = np.array(country_region_data.loc[di:di+20]).flatten()
                X_train.append(np.append(xi,[country_index,region_index,death_to_case_scale,case_death_delay,population]))
                y_train.append(np.array(country_region_data.loc[di+21:di+21+20]['rescaled_cases']))

            #Get the last 3 weeks as test
            X_test.append(X_train.pop())
            y_test.append(y_train.pop())

    return np.array(X_train), np.array(y_train),np.array(X_test), np.array(y_test)


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
    X_train = np.load(outdir+'X_train.npy', allow_pickle=True)
    y_train = np.load(outdir+'y_train.npy', allow_pickle=True)
    X_test = np.load(outdir+'X_test.npy', allow_pickle=True)
    y_test = np.load(outdir+'y_test.npy', allow_pickle=True)
except:
    sel=get_features(adjusted_data)
    X_train,y_train,X_test,y_test = split_for_training(sel)
    #Save
    np.save(outdir+'X_train.npy',X_train)
    np.save(outdir+'y_train.npy',y_train)
    np.save(outdir+'X_test.npy',X_test)
    np.save(outdir+'y_test.npy',y_test)

pdb.set_trace()
for i in range(y.shape[1]):
    reg = LinearRegression().fit(X, y[:,i])
    print(i,'score',reg.score(X, y[:,i]))
    pred = reg.predict(X)
    print('Error',np.average(np.absolute(pred-y[:,i])))
    plt.scatter(pred,y[:,i],s=1)
    plt.title(i)
    plt.savefig(outdir+str(i)+'.png',format='png')
    plt.close()

'''


'''
