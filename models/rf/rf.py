#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import sys
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.metrics import mutual_info_score
from sklearn.model_selection import RandomizedSearchCV,  GridSearchCV
from scipy.stats import pearsonr
from scipy import stats
from math import e
import _pickle as pickle
import numpy as np



import pdb
#Arguments for argparse module:
parser = argparse.ArgumentParser(description = '''Simple RF model.''')

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
parser.add_argument('--threshold', nargs=1, type= float,
                  default=sys.stdin, help = 'Threshold.')
parser.add_argument('--outdir', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to output directory. Include /in end')


def get_features(adjusted_data,train_days,forecast_days,t,outdir):
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
                        #'retail_and_recreation',
                        #'grocery_and_pharmacy',
                        #'parks',
                        #'transit_stations',
                        #'workplaces',
                        #'residential',
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
        X = np.load(outdir+'X.npy', allow_pickle=True)
        y = np.load(outdir+'y.npy', allow_pickle=True)



    except:
        sel = adjusted_data[selected_features]
        X,y = split_for_training(sel,train_days,forecast_days)

        #Save
        np.save(outdir+'X.npy',X)
        np.save(outdir+'y.npy',y)


    high_i = np.argwhere(X[:,12]>t)
    low_i = np.argwhere(X[:,12]<=t)
    X_high = X[high_i][:,0,:]
    y_high = y[high_i][:,0]
    X_low = X[low_i][:,0,:]
    y_low = y[low_i][:,0]

    #look at period differences:
    choose_uniform(X_high,y_high,500,'high')
    choose_uniform(X_low,y_low,500,'low')
    #Plot distribution
    fig,ax = plt.subplots(figsize=(6/2.54,6/2.54))
    plt.hist(np.log10(y_high+0.001),bins=20,alpha=0.75,label='high')
    plt.hist(np.log10(y_low+0.001),bins=20,alpha=0.75,label='low')
    plt.title('Target case disributions')
    plt.xlabel('log cases per 100000')
    plt.yticks([])
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir+'case_distr.png',format='png',dpi=300)
    plt.close()


    return X_high,y_high,X_low,y_low

def choose_uniform(X,y,num,mode):
    '''Function for choosing data
    '''

    prev_cases = X[:,12].copy()
    prev_cases[prev_cases<1]=1
    xy_diff = y/prev_cases
    xy_diff = np.log10(xy_diff+0.01)
    chosen_i = np.array([],dtype='int32')
    for step in np.arange(-2,2.1,0.1):
        sel = np.argwhere((xy_diff>step)&(xy_diff<=step+0.1))
        if len(sel)>num:
            sel = np.random.choice(sel.flatten(),500,replace=False)
        chosen_i = np.concatenate([chosen_i,sel.flatten()])

    X = X[chosen_i]
    y = y[chosen_i]

    prev_cases = X[:,12].copy()
    prev_cases[prev_cases<1]=1
    xy_diff = y/prev_cases
    xy_diff = np.log10(xy_diff+0.01)
    fig,ax = plt.subplots(figsize=(6/2.54,6/2.54))
    plt.hist(xy_diff,bins=50)
    plt.title(mode+' change disribution')
    plt.xlabel('log change in median cases per 100000')
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(outdir+mode+'_change_distr.png',format='png',dpi=300)
    plt.close()


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

            #Normalize the cases by 100'000 population
            #country_region_data['rescaled_cases']=country_region_data['rescaled_cases']/(population/_high000)
            #country_region_data['cumulative_rescaled_cases']=country_region_data['cumulative_rescaled_cases']/(population/_high000)
            country_region_data['smoothed_cases']=country_region_data['smoothed_cases']/(population/100000)
            country_region_data['cumulative_smoothed_cases']=country_region_data['cumulative_smoothed_cases']/(population/100000)
            #Loop through and get the data

            for di in range(len(country_region_data)-(train_days+forecast_days-1)):

                #Get all features
                xi = np.array(country_region_data.loc[di:di+train_days-1])
                #Get change over the past train days
                #period_change = xi[-1,13]-xi[0,13]
                case_medians = np.median(xi[:,12:14],axis=0)
                xi = np.average(xi,axis=0)
                xi[12:14]=case_medians


                #Normalize the cases with the input period mean
                yi = np.array(country_region_data.loc[di+train_days:di+train_days+forecast_days-1]['smoothed_cases'])
                yi = np.median(yi) #divide by average observed or total observe in period?

                #Add
                X.append(np.append(xi.flatten(),[death_to_case_scale,case_death_delay,gross_net_income,population_density,
                                                #period_change,
                                                pdi, idv, mas, uai, ltowvs, ivr,upop, pop65, gdp, obesity,
                                                cancer, smoking_deaths, pneumonia_dr, air_pollution_deaths, co2_emission,
                                                air_transport, population]))
                y.append(yi)

    return np.array(X), np.array(y)

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

def opt_model(X, y, NFOLD, mode, outdir):
    '''Fit the linear model
    '''
    #Fit the model
    param_grid = {'bootstrap': [True, False],
     'max_depth': [50, 100, None],
     'max_features': ['auto', 'sqrt'],
     'min_samples_leaf': [1, 2],
     'min_samples_split': [2, 5],
     'n_estimators': [100, 200, 300]}
    reg = RandomForestRegressor(n_jobs=-1, random_state=42)
    rf_opt = GridSearchCV(estimator = reg, param_grid = param_grid,
                        cv = NFOLD, verbose=2, n_jobs = -1)
    rf_opt.fit(X, y)

    #Write to file
    with open(outdir+mode+'_opt.txt', 'a+') as file:
        file.write("# Tuning hyper-parameters \n")
        file.write("Best parameters set found on development set:" + '\n')
        file.write(str(rf_opt.best_params_))
        file.write('\n' + '\n' + "Grid scores on development set:" + '\n')
        means =rf_opt.cv_results_['mean_test_score']
        stds = rf_opt.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, rf_opt.cv_results_['params']):
            file.write('mean test score: ')
            file.write("%0.3f (+/-%0.03f) for %r"
                  % ( mean, std * 2, params) + '\n')


    return None

def fit_model(model,X, y, NFOLD, mode, outdir):
    #KFOLD
    NFOLD = 5
    train_split, val_split = kfold(len(X),NFOLD)

    #Save errors
    mdi_importances = []
    for fold in range(NFOLD):
        tr_idx, val_idx = train_split[fold], val_split[fold]
        print('Number of valid points',len(val_idx))
        #Extract train and valid data
        X_train, y_train, X_valid, y_valid = X[tr_idx], y[tr_idx],X[val_idx], y[val_idx]
        corrs = []
        errors = []

        reg = model.fit(X_train, y_train)
        pred = reg.predict(X_valid)
        pred = pred

        #pred = np.power(e,pred)
        #if mode =='high':
        #    pred[pred>5000]=5000
        #if mode=='low':
        #    pred[pred>10]=10
        true = y_valid
        av_er = np.average(np.absolute(pred-true))

        R,p = pearsonr(pred,true)
        print('Fold',fold+1,'Average error',av_er,'PCC',R)
        plt.scatter(np.log10(true+0.001),np.log10(pred+0.001),s=1)
        plt.xlabel('True')
        plt.ylabel('Pred')
        plt.savefig(outdir+mode+str(fold)+'.png',format='png',dpi=300)
        plt.close()
        #Save
        corrs.append(R)
        errors.append(av_er)

        #Feature importances
        mdi_importances.append(reg.feature_importances_)

        #Save
        np.save(outdir+'corrs'+str(fold+1)+'.npy',np.array(corrs))
        np.save(outdir+'errors'+str(fold+1)+'.npy',np.array(errors))
        with open(outdir+'model'+str(fold), 'wb') as f:
            pickle.dump(reg, f)

    np.save(outdir+'feature_importances.npy',np.array(mdi_importances))


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
threshold = args.threshold[0]
outdir = args.outdir[0]

#Use only data from start date
adjusted_data = adjusted_data[adjusted_data['Date']>=start_date]
#Exclude the regional data from Brazil
exclude_index = adjusted_data[(adjusted_data['CountryCode']=='BRA')&(adjusted_data['RegionCode']!='0')].index
adjusted_data = adjusted_data.drop(exclude_index)

#Select only world area data
world_areas = {1:'Latin America & Caribbean', 2:'South Asia', 3:'Sub-Saharan Africa',
               4:'Europe & Central Asia', 5:'Middle East & North Africa',
               6:'East Asia & Pacific', 7:'North America'}
#adjusted_data = adjusted_data[adjusted_data['world_area']==world_areas[world_area]]
#Get data
X_high,y_high,X_low,y_low =  get_features(adjusted_data,train_days,forecast_days,threshold,outdir)

print('Number periods in high cases selection',len(y_high))
print('Number periods in low cases selection',len(y_low))


#Fit model
#opt_model(X_high,y_high,5,'high',outdir+'high/')
#opt_model(X_low,y_low,5,'low',outdir+'low/')
fit_model(RandomForestRegressor(bootstrap=True,max_depth=50,max_features='auto',
min_samples_leaf=2,min_samples_split=5, n_estimators=100, n_jobs=-1, random_state=42),
X_high,y_high,5,'high',outdir+'high/')
fit_model(RandomForestRegressor(bootstrap=False,max_depth=50,max_features='sqrt',
min_samples_leaf=2,min_samples_split=5, n_estimators=100, n_jobs=-1, random_state=42),
X_low,y_low,5,'low',outdir+'low/')
