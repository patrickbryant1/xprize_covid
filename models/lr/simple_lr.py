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
from sklearn import preprocessing
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
from scipy import stats
import numpy as np



import pdb
#Arguments for argparse module:
parser = argparse.ArgumentParser(description = '''Simple linear regression model.''')

parser.add_argument('--adjusted_data', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to processed data file.')
parser.add_argument('--start_date', nargs=1, type= str,
                  default=sys.stdin, help = 'Date to start from.')
parser.add_argument('--train_days', nargs=1, type= int,
                  default=sys.stdin, help = 'Days to include in fitting.')
parser.add_argument('--forecast_days', nargs=1, type= int,
                  default=sys.stdin, help = 'Days to forecast.')
parser.add_argument('--world_area', nargs=1, type= int,
                  default=sys.stdin, help = 'World area.')
parser.add_argument('--outdir', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to output directory. Include /in end')

def normalize_data(sel):
    '''Normalize and transform data
    '''

    # to_log = ['smoothed_cases','cumulative_smoothed_cases','rescaled_cases','cumulative_rescaled_cases','population_density', 'population']
    # for var in to_log:
    #     sel[var] = np.log10(sel[var]+0.001)

    #GNI: group into 3: 0-20k,20-40k,40k+
    index1 = sel[sel['gross_net_income']<20000].index
    above = sel[sel['gross_net_income']>20000]
    index2 = above[above['gross_net_income']<40000].index
    index3 = sel[sel['gross_net_income']>40000].index
    sel.at[index1,'gross_net_income']=0
    sel.at[index2,'gross_net_income']=1
    sel.at[index3,'gross_net_income']=2

    return sel


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
                        #'rescaled_cases',
                        #'cumulative_rescaled_cases',
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
        X100 = np.load(outdir+'X100.npy', allow_pickle=True)
        y100 = np.load(outdir+'y100.npy', allow_pickle=True)
        X_low = np.load(outdir+'X_low.npy', allow_pickle=True)
        y_low = np.load(outdir+'y_low.npy', allow_pickle=True)
        populations = np.load(outdir+'populations.npy', allow_pickle=True)
        regions = np.load(outdir+'regions.npy', allow_pickle=True)

    except:
        sel = adjusted_data[selected_features]
        #Normalize
        sel = normalize_data(sel)
        X100,y100,X_low,y_low,populations,regions = split_for_training(sel,train_days,forecast_days)

        #Save
        np.save(outdir+'X100.npy',X100)
        np.save(outdir+'y100.npy',y100)
        np.save(outdir+'X_low.npy',X_low)
        np.save(outdir+'y_low.npy',y_low)
        np.save(outdir+'populations.npy',populations)
        np.save(outdir+'regions.npy',regions)



    return X100,y100,X_low,y_low,populations,regions

def split_for_training(sel,train_days,forecast_days):
    '''Split the data for training and testing
    '''
    X100 = [] #Input periods where input/target period reaches at least 100 cases
    y100 = [] #Targets
    X_low = [] #Input periods where input/target period reaches at most 100 cases
    y_low = []

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
            #country_region_data['rescaled_cases']=country_region_data['rescaled_cases']/(population/100000)
            #country_region_data['cumulative_rescaled_cases']=country_region_data['cumulative_rescaled_cases']/(population/100000)
            #country_region_data['smoothed_cases']=country_region_data['smoothed_cases']/(population/100000)
            #country_region_data['cumulative_smoothed_cases']=country_region_data['cumulative_smoothed_cases']/(population/100000)
            #Loop through and get the data

            for di in range(len(country_region_data)-(train_days+forecast_days-1)):

                #Get all features
                xi = np.array(country_region_data.loc[di:di+train_days-1])
                #Normalize the cases with the period medians
                sm_norm = max(np.median(xi[:,12]),1)
                sm_cum_norm = max(np.median(xi[:,13]),1)
                xi[:,12]=xi[:,12]/sm_norm
                xi[:,13]=xi[:,13]/sm_cum_norm
                #Get change over the past train days
                period_change = xi[-1,13]-xi[0,13]
                #Get targets
                yi = np.array(country_region_data.loc[di+train_days:di+train_days+forecast_days-1]['smoothed_cases'])
                yi = yi/sm_norm

                #Add
                #Check the highest daily cases in the period
                if max(country_region_data.loc[di:di+train_days-1,'smoothed_cases'])>100:
                    X100.append(np.append(xi.flatten(),[death_to_case_scale,case_death_delay,gross_net_income,population_density,period_change,pdi, idv, mas, uai, ltowvs, ivr, population]))
                    y100.append(yi)
                else:
                    X_low.append(np.append(xi.flatten(),[death_to_case_scale,case_death_delay,gross_net_income,population_density,period_change,pdi, idv, mas, uai, ltowvs, ivr, population]))
                    y_low.append(yi)

            #Save population
            populations.append(population)

    return np.array(X100), np.array(y100),np.array(X_low), np.array(y_low), np.array(populations), np.array(regions)

def kfold(num_regions, NFOLD):
    '''Generate a K-fold split using numpy (can't import sklearn everywhere)
    '''
    all_i = np.arange(num_regions)
    train_split = []
    val_split = []
    fetched_i = []
    #Check
    check = np.zeros(num_regions)
    #Go through ll folds
    for f in range(NFOLD):
        remaining_i = np.setdiff1d(all_i,np.array(fetched_i))
        val_i = np.random.choice(remaining_i,int(num_regions/NFOLD),replace=False)
        train_i = np.setdiff1d(all_i,val_i)
        #Save
        val_split.append(val_i)
        train_split.append(train_i)
        fetched_i.extend(val_i)
        check[val_i]+=1

    return np.array(train_split), np.array(val_split)

def fit_model(X, y, NFOLD, outdir):
    '''Fit the linear model
    '''
    #Fit the model

    #KFOLD
    NFOLD = 5
    #kf = KFold(n_splits=NFOLD,shuffle=True, random_state=42)
    train_split, val_split = kfold(len(X),NFOLD)

    #Save errors
    train_errors = []
    valid_errors = []
    for fold in range(NFOLD):
        tr_idx, val_idx = train_split[fold], val_split[fold]
        print('Number of valid points',len(val_idx))
        #Extract train and valid data
        X_train, y_train, X_valid, y_valid = X[tr_idx], y[tr_idx],X[val_idx], y[val_idx]
        corrs = []
        errors = []
        coefs = []
        intercepts = []
        #Fit each day
        for day in range(y_train.shape[1]):
            reg = LinearRegression().fit(X_train, y_train[:,day])
            pred = reg.predict(X_valid)

            #Ensure non-negative
            pred[pred<0]=0
            true = y_valid[:,day]
            av_er = np.average(np.absolute(pred-true))

            R,p = pearsonr(pred,true)
            print('Fold',fold+1,'Day',day,'Average error',av_er,'PCC',R)
            #Save
            corrs.append(R)
            errors.append(av_er)
            coefs.append(reg.coef_)
            intercepts.append(reg.intercept_)


        #Save
        np.save(outdir+'corrs'+str(fold+1)+'.npy',np.array(corrs))
        np.save(outdir+'errors'+str(fold+1)+'.npy',np.array(errors))
        np.save(outdir+'coefs'+str(fold+1)+'.npy',np.array(coefs))
        np.save(outdir+'intercepts'+str(fold+1)+'.npy',np.array(intercepts))
    pdb.set_trace()
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
world_area = args.world_area[0]
outdir = args.outdir[0]

#Use only data from start date
adjusted_data = adjusted_data[adjusted_data['Date']>=start_date]
#Select only world area data
world_areas = {1:"Europe & Central Asia"}
#adjusted_data = adjusted_data[adjusted_data['world_area']==world_areas[world_area]]

#Get data
X100,y100,X_low,y_low,populations,regions =  get_features(adjusted_data,train_days,forecast_days,outdir)
print('Number periods in above 100 cases selection',len(y100))
print('Number periods in below 100 cases selection',len(y_low))
#Fit model
fit_model(X100,y100,5,outdir+'above100/')
fit_model(X_low,y_low,5,outdir+'low/')
