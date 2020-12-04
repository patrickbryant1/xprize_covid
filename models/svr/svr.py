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
from sklearn.svm import LinearSVR
from sklearn.model_selection import KFold
import numpy as np


import pdb
#Arguments for argparse module:
parser = argparse.ArgumentParser(description = '''Support vector regression model with linear kernel.''')

parser.add_argument('--adjusted_data', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to processed data file.')
parser.add_argument('--start_date', nargs=1, type= str,
                  default=sys.stdin, help = 'Date to start from.')
parser.add_argument('--train_days', nargs=1, type= int,
                  default=sys.stdin, help = 'Days to include in fitting.')
parser.add_argument('--forecast_days', nargs=1, type= int,
                  default=sys.stdin, help = 'Days to forecast.')
parser.add_argument('--outdir', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to output directory. Include /in end')


def get_features(adjusted_data,train_days,forecast_days,outdir):
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
                        'CountryName',
                        'RegionName',
                        'smoothed_cases',
                        'cumulative_smoothed_cases',
                        'rescaled_cases',
                        'cumulative_rescaled_cases',
                        'death_to_case_scale',
                        'case_death_delay',
                        'gross_net_income',
                        'population_density',
                        'monthly_temperature',
                        'retail_and_recreation',
                        'grocery_and_pharmacy',
                        'parks',
                        'transit_stations',
                        'workplaces',
                        'residential',
                        'pdi', 'idv', 'mas', 'uai', 'ltowvs', 'ivr',
                        'population']

    #Get features
    try:
        X = np.load(outdir+'X.npy', allow_pickle=True)
        y = np.load(outdir+'y.npy', allow_pickle=True)
        populations = np.load(outdir+'populations.npy', allow_pickle=True)
        regions = np.load(outdir+'regions.npy', allow_pickle=True)

    except:
        sel = adjusted_data[selected_features]
        X,y,populations,regions = split_for_training(sel,train_days,forecast_days)
        #Save
        np.save(outdir+'X.npy',X)
        np.save(outdir+'y.npy',y)
        np.save(outdir+'populations.npy',populations)
        np.save(outdir+'regions.npy',regions)



    return X,y,populations,regions

def split_for_training(sel,train_days,forecast_days):
    '''Split the data for training and testing
    '''
    X = [] #Inputs
    y = [] #Targets

    countries = sel['Country_index'].unique()
    populations = []
    regions = []
    for ci in countries:
        country_data = sel[sel['Country_index']==ci]
        #Check regions
        country_regions = country_data['Region_index'].unique()
        for ri in country_regions:
            country_region_data = country_data[country_data['Region_index']==ri]
            #Select data 14 days before above 0 cases
            try:
                si = max(0,country_region_data[country_region_data['cumulative_smoothed_cases']>0].index[0]-14)
                country_region_data = country_region_data.loc[si:]
            except:
                print(len(country_region_data[country_region_data['cumulative_smoothed_cases']>0]),'cases for',country_region_data['CountryName'].unique()[0])
                continue

            country_region_data = country_region_data.reset_index()

            #Check if data
            if len(country_region_data)<train_days+forecast_days+1:
                print('Not enough data for',country_region_data['CountryName'].values[0])
                continue

            country_index = country_region_data.loc[0,'Country_index']
            region_index = country_region_data.loc[0,'Region_index']
            death_to_case_scale = country_region_data.loc[0,'death_to_case_scale']
            case_death_delay = country_region_data.loc[0,'case_death_delay']
            gross_net_income = country_region_data.loc[0,'gross_net_income']
            population_density = country_region_data.loc[0,'population_density']
            pdi = country_region_data.loc[0,'pdi'] #Power distance
            idv = country_region_data.loc[0, 'idv'] #Individualism
            mas = country_region_data.loc[0,'mas'] #Masculinity
            uai = country_region_data.loc[0,'uai'] #Uncertainty
            ltowvs = country_region_data.loc[0,'ltowvs'] #Long term orientation,  describes how every society has to maintain some links with its own past while dealing with the challenges of the present and future
            ivr = country_region_data.loc[0,'ivr'] #Indulgence, Relatively weak control is called “Indulgence” and relatively strong control is called “Restraint”.
            #PC1 =  country_region_data.loc[0,'PC1'] #Principal components 1 and 2 of last 90 days of cases
            #PC2 =  country_region_data.loc[0,'PC2']
            population = country_region_data.loc[0,'population']
            if region_index!=0:
                regions.append(country_region_data.loc[0,'CountryName']+'_'+country_region_data.loc[0,'RegionName'])
            else:
                regions.append(country_region_data.loc[0,'CountryName'])

            country_region_data = country_region_data.drop(columns={'index','Country_index', 'Region_index','CountryName',
            'RegionName', 'death_to_case_scale', 'case_death_delay', 'gross_net_income','population_density','pdi', 'idv',
             'mas', 'uai', 'ltowvs', 'ivr','population'})

            #Normalize the cases by 100'000 population
            country_region_data['rescaled_cases']=country_region_data['rescaled_cases']/(population/100000)
            country_region_data['cumulative_rescaled_cases']=country_region_data['cumulative_rescaled_cases']/(population/100000)
            country_region_data['smoothed_cases']=country_region_data['smoothed_cases']/(population/100000)
            country_region_data['cumulative_smoothed_cases']=country_region_data['cumulative_smoothed_cases']/(population/100000)
            #Loop through and get the data
            X_region = []
            y_region = []
            for di in range(len(country_region_data)-(train_days+forecast_days-1)):
                #Get change over the past 21 days
                xi = np.array(country_region_data.loc[di:di+train_days-1])[:,13]
                period_change = xi[-1]-xi[0]
                #Add
                X_region.append(np.append(xi[0], [xi[-1],period_change])) #[country_index,region_index,death_to_case_scale,case_death_delay,gross_net_income,population_density,period_change,pdi, idv, mas, uai, ltowvs, ivr, population]))
                y_region.append(np.array(country_region_data.loc[di+train_days:di+train_days+forecast_days-1]['smoothed_cases']))

            #Save X and y for region
            X.append(np.array(X_region))
            y.append(np.array(y_region))
            #Save population
            populations.append(population)

    return np.array(X), np.array(y), np.array(populations), np.array(regions)



def fit_model(X, y, NFOLD, outdir):
    '''Fit the linear model
    '''
    #Fit the model

    #KFOLD
    kf = KFold(n_splits=NFOLD, random_state=42)
    #Perform K-fold CV
    FOLD=0
    for tr_idx, val_idx in kf.split(X):

        FOLD+=1
        X_train, y_train, X_valid, y_valid = X[tr_idx], y[tr_idx], X[val_idx], y[val_idx]
        corrs = []
        errors = []
        coefs = []
        intercepts = []
        #Extract train and valid data by country region
        X_train_extracted = X_train[0]
        y_train_extracted = y_train[0]

        for cr in range(1,len(X_train)):
            X_train_extracted = np.append(X_train_extracted,X_train[cr],axis=0)
            y_train_extracted = np.append(y_train_extracted,y_train[cr],axis=0)

        X_valid_extracted = X_valid[0]
        y_valid_extracted = y_valid[0]
        for cr in range(1,len(X_valid)):
            X_valid_extracted = np.append(X_valid_extracted,X_valid[cr],axis=0)
            y_valid_extracted = np.append(y_valid_extracted,y_valid[cr],axis=0)

        #Fit each day
        for day in range(y_train[0].shape[1]):
            reg = LinearSVR(max_iter=100000).fit(X_train_extracted, y_train_extracted[:,day])
            pred = reg.predict(X_valid_extracted)

            #Ensure non-negative
            pred[pred<0]=0
            true = y_valid_extracted[:,day]
            av_er = np.average(np.absolute(pred-true))

            R,p = pearsonr(pred,true)
            print('Fold',FOLD,'Day',day,'Average error',av_er,'PCC',R)
            #Save
            corrs.append(R)
            errors.append(av_er)
            coefs.append(reg.coef_)
            intercepts.append(reg.intercept_)


        #Save
        np.save(outdir+'corrs'+str(FOLD)+'.npy',np.array(corrs))
        np.save(outdir+'errors'+str(FOLD)+'.npy',np.array(errors))
        np.save(outdir+'coefs'+str(FOLD)+'.npy',np.array(coefs))
        np.save(outdir+'intercepts'+str(FOLD)+'.npy',np.array(intercepts))

    return None



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
start_date = args.start_date[0]
train_days = args.train_days[0]
forecast_days = args.forecast_days[0]
outdir = args.outdir[0]

#Use only data from start date
adjusted_data = adjusted_data[adjusted_data['Date']>=start_date]

#Get data
X,y,populations,regions =  get_features(adjusted_data,train_days,forecast_days,outdir)

#Fit model
corrs, errors, coefs, intercepts = fit_model(X,y,5,outdir)
