#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import sys
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression,Ridge,ElasticNet
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
            if len(country_region_data)<1:
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
            for di in range(len(country_region_data)-(train_days+forecast_days-1)):
                #Get change over the past 21 days
                xi = np.array(country_region_data.loc[di:di+train_days-1]).flatten()
                period_change = xi[-country_region_data.shape[1]:][13]-xi[:country_region_data.shape[1]][13]
                #Add
                X.append(np.append(xi,[country_index,region_index,death_to_case_scale,case_death_delay,gross_net_income,population_density,period_change,pdi, idv, mas, uai, ltowvs, ivr, population]))
                y.append(np.array(country_region_data.loc[di+train_days:di+train_days+forecast_days-1]['smoothed_cases']))

            #Save population
            populations.append(population)

    return np.array(X), np.array(y), np.array(populations), np.array(regions)

def fit_model(X, y, NFOLD, outdir):
    '''Fit the linear model
    '''
    try:
        #If the model has already been fitted
        corrs = np.load(outdir+'corrs.npy',allow_pickle=True)
        errors = np.load(outdir+'errors.npy',allow_pickle=True)
        coefs = np.load(outdir+'coefs.npy',allow_pickle=True)
    except:
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
            stds = []
            coefs = []
            for day in range(y_train.shape[1]):
                reg = LinearRegression().fit(X_train, y_train[:,day])
                pred = reg.predict(X_valid)
                #No negative predictions are allowed
                pred[pred<0]=0
                av_er = np.average(np.absolute(pred-y_valid[:,day]))
                print('Fold',FOLD,'Day',day,'Average error',av_er)
                R,p = pearsonr(pred,y_valid[:,day])
                #Save
                corrs.append(R)
                errors.append(av_er)
                coefs.append(reg.coef_)

            #Save
            np.save(outdir+'corrs'+str(FOLD)+'.npy',np.array(corrs))
            np.save(outdir+'errors'+str(FOLD)+'.npy',np.array(errors))
            np.save(outdir+'coefs'+str(FOLD)+'.npy',np.array(coefs))
        pdb.set_trace()

    return corrs, errors, stds, preds, coefs

def evaluate_model(corrs, errors, stds, preds, coefs, y_test, train_days,outdir):
    '''Evaluate the fit model
    '''

    #Evaluate model
    results_file = open(outdir+'results.txt','w')
    #Calculate error
    results_file.write('Total 2week mae: '+str(np.sum(total_regional_2week_mae))+'\n')
    results_file.write('Total mae: '+str(np.sum(total_regional_mae))+'\n')
    results_file.write('Total mae per 100000: '+str(np.sum(total_regional_mae_per_100000))+'\n')
    results_file.write('Total cumulative error: '+str(np.sum(total_regional_cum_error))+'\n')

    #Look at coefs
    #The first are repeats 21 times, then single_features follow: [country_index,region_index,death_to_case_scale,case_death_delay,gross_net_income,population_density,population]
    #--> get the last features, then divide into 21 portions

    single_feature_names=['country_index','region_index','death_to_case_scale','case_death_delay','gross_net_income','population_density','Change in input period days','pdi', 'idv', 'mas', 'uai', 'ltowvs', 'ivr','population']
    #days pred,days behind - this goes from -21 to 1,features
    repeat_feature_names = ['C1_School closing', 'C2_Workplace closing', 'C3_Cancel public events', 'C4_Restrictions on gatherings', 'C5_Close public transport', 'C6_Stay at home requirements',
    'C7_Restrictions on internal movement', 'C8_International travel controls', 'H1_Public information campaigns', 'H2_Testing policy', 'H3_Contact tracing', 'H6_Facial Coverings',
    'smoothed_cases', 'cumulative_smoothed_cases', 'rescaled_cases', 'cumulative_rescaled_cases', 'monthly_temperature', 'retail_and_recreation', 'grocery_and_pharmacy', 'parks','transit_stations', 'workplaces', 'residential']
    all_feature_names = single_feature_names+repeat_feature_names*train_days
    for i in range(coefs.shape[0]):
        fig,ax=plt.subplots(figsize=(18,6))
        plt.bar(range(coefs.shape[1]),coefs[i,:],)
        for j in range(coefs.shape[1]):
            plt.text(j,coefs[i,j],all_feature_names[j],fontsize=12)

        plt.title('Day '+str(i+1))
        plt.tight_layout()
        plt.savefig(outdir+'coefs_'+str(i+1)+'.png',format='png',dpi=300)
        plt.close()

    #Plot average error per day with std
    plt.plot(range(1,22),errors,color='b')
    plt.fill_between(range(1,22),errors-stds,errors+stds,color='b',alpha=0.5)
    plt.title('Average error with std')
    plt.xlabel('Days in the future')
    plt.ylabel('Error per 100000')
    plt.savefig(outdir+'lr_av_error.png',format='png')
    plt.close()

    #Plot correlation
    plt.plot(range(1,22),corrs ,color='b')
    plt.title('Pearson R')
    plt.xlabel('Days in the future')
    plt.ylabel('PCC')
    plt.savefig(outdir+'PCC.png',format='png')
    plt.close()

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
corrs, errors, stds, preds, coefs = fit_model(X,y,5,outdir)
#Evaluate model
evaluate_model(corrs, errors, stds, preds, coefs, y_test, train_days,outdir)
