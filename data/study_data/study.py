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
import seaborn as sns
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
                        'Urban population (% of total population)',
                        'Population ages 65 and above (% of total population)',
                        'GDP per capita (current US$)', 'Obesity Rate (%)', 'Cancer Rate (%)',
                        'Share of Deaths from Smoking (%)', 'Pneumonia Death Rate (per 100K)',
                        'Share of Deaths from Air Pollution (%)',
                        'CO2 emissions (metric tons per capita)',
                        'Air transport (# carrier departures worldwide)',
                        'population']

    #Get features
    try:
        X = np.load(outdir+'X_'+str(train_days)+'_'+str(forecast_days)+'.npy', allow_pickle=True)
        y = np.load(outdir+'y_'+str(train_days)+'_'+str(forecast_days)+'.npy', allow_pickle=True)
        day_closest_to_target = np.load(outdir+'day_closest_to_target_'+str(train_days)+'_'+str(forecast_days)+'.npy', allow_pickle=True)

    except:
        sel = adjusted_data[selected_features]
        X,y,day_closest_to_target = split_for_training(sel,train_days,forecast_days)

        #Save
        np.save(outdir+'X_'+str(train_days)+'_'+str(forecast_days)+'.npy',X)
        np.save(outdir+'y_'+str(train_days)+'_'+str(forecast_days)+'.npy',y)
        np.save(outdir+'day_closest_to_target_'+str(train_days)+'_'+str(forecast_days)+'.npy',day_closest_to_target)

    return X,y,day_closest_to_target

def split_for_training(sel,train_days,forecast_days):
    '''Split the data for training and testing
    '''
    X = [] #Input periods
    y = [] #Targets
    day_closest_to_target = [] #which future day that is the closest to the target median


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
            upop = country_region_data.loc[0,'Urban population (% of total population)']
            pop65 = country_region_data.loc[0,'Population ages 65 and above (% of total population)']
            gdp = country_region_data.loc[0,'GDP per capita (current US$)']
            obesity = country_region_data.loc[0,'Obesity Rate (%)']
            cancer = country_region_data.loc[0,'Cancer Rate (%)']
            smoking_deaths = country_region_data.loc[0,'Share of Deaths from Smoking (%)']
            pneumonia_dr = country_region_data.loc[0,'Pneumonia Death Rate (per 100K)']
            air_pollution_deaths = country_region_data.loc[0,'Share of Deaths from Air Pollution (%)']
            co2_emission = country_region_data.loc[0,'CO2 emissions (metric tons per capita)']
            air_transport = country_region_data.loc[0,'Air transport (# carrier departures worldwide)']
            population = country_region_data.loc[0,'population']
            if region_index!=0:
                regions.append(country_region_data.loc[0,'CountryName']+'_'+country_region_data.loc[0,'RegionName'])
            else:
                regions.append(country_region_data.loc[0,'CountryName'])

            country_region_data = country_region_data.drop(columns={'index','Country_index', 'Region_index','CountryName',
            'RegionName', 'death_to_case_scale', 'case_death_delay', 'gross_net_income','population_density','pdi', 'idv',
             'mas', 'uai', 'ltowvs', 'ivr','Urban population (% of total population)','Population ages 65 and above (% of total population)',
             'GDP per capita (current US$)', 'Obesity Rate (%)', 'Cancer Rate (%)', 'Share of Deaths from Smoking (%)', 'Pneumonia Death Rate (per 100K)',
             'Share of Deaths from Air Pollution (%)','CO2 emissions (metric tons per capita)', 'Air transport (# carrier departures worldwide)','population'})

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

                #Get median
                yi = np.array(country_region_data.loc[di+train_days:di+train_days+forecast_days-1]['smoothed_cases'])
                median_yi = np.median(yi)
                #look at which day is the closest to the median
                median_diff = np.absolute(yi-median_yi)
                day_closest_to_target.append(np.average(np.argwhere(median_diff==min(median_diff))))

                #Add
                X.append(np.append(xi.flatten(),[death_to_case_scale,case_death_delay,gross_net_income,population_density,
                                                period_change,pdi, idv, mas, uai, ltowvs, ivr,upop, pop65, gdp, obesity,
                                                cancer, smoking_deaths, pneumonia_dr, air_pollution_deaths, co2_emission,
                                                air_transport, population]))
                y.append(median_yi)


    return np.array(X), np.array(y), np.array(day_closest_to_target)

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


def feature_outcome(X_high,y_high,dct_high,X_low,y_low,dct_low,outdir):
    '''Study the difference in outcome due to a feture being of a certain kind
    '''

    #Look at dct vs y
    fig,ax = plt.subplots(figsize=(4.5/2.54,4.5/2.54))
    plt.scatter(dct_low+1,np.log10(y_low+0.001),s=0.1)
    plt.tight_layout()
    plt.savefig(outdir+'dct_low.png',format='png',dpi=300)
    plt.close()

    fig,ax = plt.subplots(figsize=(4.5/2.54,4.5/2.54))
    plt.scatter(dct_high+1,np.log10(y_high+0.001),s=0.1)
    plt.tight_layout()
    plt.savefig(outdir+'dct_high.png',format='png',dpi=300)
    plt.close()

    fig,ax = plt.subplots(figsize=(4.5/2.54,4.5/2.54))
    plt.hist(dct_low+1,alpha=0.5,label='low',bins=21)
    plt.hist(dct_high+1,alpha=0.5,label='high',bins=21)
    plt.yticks([])
    plt.legend()
    plt.xlabel('Day of median')
    plt.tight_layout()
    plt.savefig(outdir+'dct_distr.png',format='png',dpi=300)
    plt.close()
    pdb.set_trace()


    feature_names = ['C1_School closing','C2_Workplace closing','C3_Cancel public events',
                    'C4_Restrictions on gatherings', 'C5_Close public transport', 'C6_Stay at home requirements',
                    'C7_Restrictions on internal movement','C8_International travel controls','H1_Public information campaigns',
                    'H2_Testing policy','H3_Contact tracing','H6_Facial Coverings', 'smoothed_cases','cumulative_smoothed_cases',
                    'death_to_case_scale', 'case_death_delay', 'gross_net_income','population_density', 'monthly_temperature',
                    'retail_and_recreation', 'grocery_and_pharmacy', 'parks', 'transit_stations', 'workplaces','residential',
                    'pdi', 'idv', 'mas', 'uai', 'ltowvs', 'ivr', 'Urban population (% of total population)','Population ages 65 and above (% of total population)',
                    'GDP per capita (current US$)', 'Obesity Rate (%)', 'Cancer Rate (%)', 'Share of Deaths from Smoking (%)', 'Pneumonia Death Rate (per 100K)',
                    'Share of Deaths from Air Pollution (%)','CO2 emissions (metric tons per capita)','Air transport (# carrier departures worldwide)','population']

    for i in range(len(feature_names)):
        fig,ax = plt.subplots(figsize=(4.5/2.54,4.5/2.54))
        plt.scatter(X_high[:,i],np.log10(y_high+0.01),color='cornflowerblue',s=1,label='high')
        plt.scatter(X_low[:,i],np.log10(y_low+0.01),color='mediumseagreen',s=1, label='low')
        plt.xlabel(feature_names[i])
        plt.ylabel('log cases per 100000')
        plt.title(feature_names[i])
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir+feature_names[i]+'.png',format='png',dpi=300)
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
            X,y,day_closest_to_target =  get_features(adjusted_data,td,fd,outdir)

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


#Look at features vs selected t from the selected periods
X,y,day_closest_to_target =  get_features(adjusted_data,21,21,outdir)
t=1.8
high_i = np.argwhere(X[:,12]>t)
low_i = np.argwhere(X[:,12]<=t)
#Select
X_high = X[high_i][:,0,:]
y_high = y[high_i][:,0]
dct_high = day_closest_to_target[high_i][:,0]
X_low = X[low_i][:,0,:]
y_low = y[low_i][:,0]
dct_low = day_closest_to_target[low_i][:,0]

feature_outcome(X_high,y_high,dct_high,X_low,y_low,dct_low,outdir)
