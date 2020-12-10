#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import sys
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.metrics import mutual_info_score
from scipy.stats import pearsonr
from scipy import stats
import numpy as np



import pdb
#Arguments for argparse module:
parser = argparse.ArgumentParser(description = '''Stude data relationships.''')

parser.add_argument('--adjusted_data', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to processed data file.')
parser.add_argument('--start_date', nargs=1, type= str,
                  default=sys.stdin, help = 'Date to start from.')
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
        X = np.load(outdir+'X_'+str(train_days)+'_'+str(forecast_days)+'.npy', allow_pickle=True)
        y = np.load(outdir+'y_'+str(train_days)+'_'+str(forecast_days)+'.npy', allow_pickle=True)

    except:
        sel = adjusted_data[selected_features]
        X,y = split_for_training(sel,train_days,forecast_days)

        #Save
        np.save(outdir+'X_'+str(train_days)+'_'+str(forecast_days)+'.npy',X)
        np.save(outdir+'y_'+str(train_days)+'_'+str(forecast_days)+'.npy',y)


    return X,y

def split_for_training(sel,train_days,forecast_days):
    '''Split the data for training and testing
    '''
    X = [] #Input periods
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

            #Normalize the cases by _high'000 population
            #country_region_data['rescaled_cases']=country_region_data['rescaled_cases']/(population/_high000)
            #country_region_data['cumulative_rescaled_cases']=country_region_data['cumulative_rescaled_cases']/(population/_high000)
            country_region_data['smoothed_cases']=country_region_data['smoothed_cases']/(population/100000)
            country_region_data['cumulative_smoothed_cases']=country_region_data['cumulative_smoothed_cases']/(population/100000)
            #Loop through and get the data

            for di in range(len(country_region_data)-(train_days+forecast_days-1)):

                #Get all features
                xi = np.array(country_region_data.loc[di:di+train_days-1])
                #Get change over the past train days
                period_change = xi[-1,13]-xi[0,13]
                xi = np.median(xi,axis=0)

                #Normalize the cases with the input period mean
                yi = np.array(country_region_data.loc[di+train_days:di+train_days+forecast_days-1]['smoothed_cases'])
                yi = np.median(yi) #divide by average observed or total observe in period?

                #Add
                X.append(np.append(xi.flatten(),[death_to_case_scale,case_death_delay,gross_net_income,population_density,period_change,pdi, idv, mas, uai, ltowvs, ivr, population]))
                y.append(yi)

    return np.array(X), np.array(y)

def calc_mutual_info(y_high,y_low):
    '''Calculate the MI by bootstrapping n random samples 10 times
    '''
    mutual_info = []
    n = min(5000,y_low.shape[0])
    for i in range(10):
        chosen_h = np.random.choice(y_high.shape[0],n,replace=False)
        chosen_l = np.random.choice(y_low.shape[0],n,replace=False)
        mutual_info.append(mutual_info_score(y_high[chosen_h],y_low[chosen_l]))

    return np.average(mutual_info), np.std(mutual_info)


# def feature_outcome(X,y):
#     '''Study the difference in outcome due to a feture being of a certain kind
#     '''

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
outdir = args.outdir[0]

#Use only data from start date
adjusted_data = adjusted_data[adjusted_data['Date']>=start_date]

try:
    avs = np.load(outdir+'case_avs.npy',allow_pickle=True)
    stds = np.load(outdir+'case_stds.npy',allow_pickle=True)
except:
    #Get data
    train_days = []
    forecast_days = []
    MI = {'case_av':[],'case_std':[]}
    for td in [7,14,21]:
        for fd in [7,14,21]:
            X,y =  get_features(adjusted_data,td,fd,outdir)

            #Investigate t vs MI
            mi_scores = []
            mi_stds = []
            for t in np.arange(0.1,10,0.1):
                y_high = y[np.argwhere(X[:,12]>t)][:,0]
                y_low = y[np.argwhere(X[:,12]<=t)][:,0]
                mi_score, mi_std = calc_mutual_info(y_high,y_low)
                mi_scores.append(mi_score)
                mi_stds.append(mi_std)
            #Save
            train_days.append(td)
            forecast_days.append(fd)
            #Save MIs
            MI['case_av'].append(np.array(mi_scores))
            MI['case_std'].append(np.array(mi_stds))

    #Save
    avs = np.array(MI['case_av'])
    stds = np.array(MI['case_std'])
    np.save(outdir+'case_avs.npy',avs)
    np.save(outdir+'case_stds.npy',stds)


#Plot
#MI
fig,ax = plt.subplots(figsize=(4.5,4.5))

colors = ['tab:blue','tab:orange','tab:green','tab:purple','tab:brown','magenta',
        'tab:gray','tab:olive','tab:cyan']
i=0
for td in [7,14,21]:
    for fd in [7,14,21]:
        plt.plot(np.arange(0.1,10,0.1),avs[i,:],color=colors[i],label=str(td)+'_'+str(fd))
        plt.fill_between(np.arange(0.1,10,0.1),avs[i,:]-stds[i,:],avs[i,:]+stds[i,:],alpha=0.5,color=colors[i])
        #plt.text(9.9,avs[i,-1],str(td)+'_'+str(fd))
        i+=1
plt.title('Threshold vs MI')
plt.xlabel('Cases per 100000')
plt.ylabel('MI')
plt.ylim([6.5,8])
plt.xticks(np.arange(0,11,1))
plt.legend()
plt.tight_layout()
plt.savefig(outdir+'t_vs_MI.png',format='png',dpi=300)
plt.close()
pdb.set_trace()
