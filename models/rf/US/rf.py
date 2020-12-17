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
from scipy.stats import pearsonr
from scipy import stats
from math import e
import _pickle as pickle
import numpy as np



import pdb
#Arguments for argparse module:
parser = argparse.ArgumentParser(description = '''Simple RF model specific for the USA.''')

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
parser.add_argument('--sex_eth_age_data', nargs=1, type= str,
                default=sys.stdin, help = 'Path to data with sex age and ethnicity per state (csv).')
parser.add_argument('--outdir', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to output directory. Include /in end')


def format_age_per_ethnicity(sex_eth_age_data):
    '''Extract ethnicity data per state
    '''
    extracted_data = pd.DataFrame()
    sexes = {0:'Total', 1:'Male', 2:'Female'}
    #Use only total
    sex_eth_age_data = sex_eth_age_data[sex_eth_age_data['SEX']==0]
    origins = {1:'Non-Hispanic',2:'Hispanic'}
    ethnicities = {1:'White', 2:'Black',3:'American Indian or Alaska Native',4:'Asian',5:'Native Hawaiian or Other Pacific Islander',6:'More than one race'}
    #Combining origin 1 and ethnicity gives Non-Hispanic + ethinicity (same used in COVID reportings)
    #AGE is single-year of age (0, 1, 2,... 84, 85+ years)
    age_groups = {0:'Under 1 year',4:'1-4 years',14:'5-14 years',24:'15-24 years',34:'25-34 years',44:'35-44 years',
                54:'45-54 years', 64:'55-64 years', 74:'65-74 years',84:'75-84 years', 85:'85 years and over'}
    age_index = [1,4,14,24,34,44,54,64,74,84]
    #Assign columns
    extracted_data['State'] = ''
    extracted_data['Ethnicity'] = ''
    for age in age_groups:
        extracted_data[age_groups[age]] = 0

    for state in sex_eth_age_data['NAME'].unique():
        per_state = sex_eth_age_data[sex_eth_age_data['NAME']==state]
        #Loop through origins
        hispanic_state = per_state[per_state['ORIGIN']==2]
        vals = []
        vals.append(state)
        vals.append('Hispanic or Latino')
        age_counts = np.zeros(len(age_groups))

        #All the ethnicities in Hispanic should be summed
        for eth in ethnicities:
            hispanic_state_eth = hispanic_state[hispanic_state['RACE']==eth] #All the Hispanic ethnicities will be summed
            hispanic_state_eth = hispanic_state_eth.reset_index()
            #First
            age_counts[0]+=np.sum(hispanic_state_eth.loc[0,'POPESTIMATE2019'])
            for i in range(len(age_index)-1):
                age_counts[i+1]+=np.sum(hispanic_state_eth.loc[age_index[i]:age_index[i+1],'POPESTIMATE2019'])
            #Last
            age_counts[-1]+=np.sum(hispanic_state_eth.loc[85:,'POPESTIMATE2019'])
        vals.extend(age_counts)
        slice = pd.DataFrame([vals],columns=extracted_data.columns)
        extracted_data=pd.concat([extracted_data,slice])

        #All the ethnicities in the NON Hispanic should NOT be summed
        #Loop through origins
        non_hispanic_state = per_state[per_state['ORIGIN']==1]

        for eth in ethnicities:
            vals = []
            vals.append(state)
            vals.append('Non-Hispanic '+ethnicities[eth])
            age_counts = np.zeros(len(age_groups))

            non_hispanic_state_eth = non_hispanic_state[non_hispanic_state['RACE']==eth] #All the Hispanic ethnicities will be summed
            non_hispanic_state_eth =non_hispanic_state_eth.reset_index()
            #First
            age_counts[0]+=np.sum(non_hispanic_state_eth.loc[0,'POPESTIMATE2019'])
            for i in range(len(age_index)-1):
                age_counts[i+1]+=np.sum(non_hispanic_state_eth.loc[age_index[i]:age_index[i+1],'POPESTIMATE2019'])
            #Last
            age_counts[-1]+=np.sum(non_hispanic_state_eth.loc[85:,'POPESTIMATE2019'])
            vals.extend(age_counts)
            slice = pd.DataFrame([vals],columns=extracted_data.columns)
            extracted_data=pd.concat([extracted_data,slice])

    #Extract all ethnicities and sums for the model
    model_formatted_data = pd.DataFrame()
    age_cols = ['Under 1 year', '1-4 years',
       '5-14 years', '15-24 years', '25-34 years', '35-44 years',
       '45-54 years', '55-64 years', '65-74 years', '75-84 years',
       '85 years and over']
    for state in extracted_data['State'].unique():
        state_data = extracted_data[extracted_data['State']==state]
        for eth in [*ethnicities.values()]:
            eth_state_data = state_data[state_data['Ethnicity']==eth]
            pdb.set_trace()
            eth_state_sum = eth_state_data[age_cols]

    #Save df
    pdb.set_trace()
    extracted_data = extracted_data.reset_index()
    extracted_data.to_csv('formatted_eth_age_data_per_state.csv')
    return extracted_data

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

def fit_model(X, y, NFOLD, mode, outdir):
    '''Fit the linear model
    '''
    #Fit the model

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

        reg = RandomForestRegressor(n_jobs=-1, random_state=42).fit(X_train, y_train)
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
sex_eth_age_data=pd.read_csv(args.sex_eth_age_data[0])
outdir = args.outdir[0]

#Format the sex_eth_age_data
format_age_per_ethnicity(sex_eth_age_data)
#Use only data from start date
adjusted_data = adjusted_data[adjusted_data['Date']>=start_date]
#Select only the USA data
adjusted_data = adjusted_data[adjusted_data['CountryCode']=='USA']
adjusted_data = adjusted_data.reset_index()
#Get data
X_high,y_high,X_low,y_low =  get_features(adjusted_data,train_days,forecast_days,threshold,outdir)

print('Number periods in high cases selection',len(y_high))
print('Number periods in low cases selection',len(y_low))

#Fit model
fit_model(X_high,y_high,5,'high',outdir+'high/')
fit_model(X_low,y_low,5,'low',outdir+'low/')
