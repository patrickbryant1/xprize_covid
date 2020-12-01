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
parser.add_argument('--outdir', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to output directory. Include /in end')


def get_features(adjusted_data,outdir):
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
        X_train = np.load(outdir+'X_train.npy', allow_pickle=True)
        y_train = np.load(outdir+'y_train.npy', allow_pickle=True)
        X_test = np.load(outdir+'X_test.npy', allow_pickle=True)
        y_test = np.load(outdir+'y_test.npy', allow_pickle=True)
        populations = np.load(outdir+'populations.npy', allow_pickle=True)
        regions = np.load(outdir+'regions.npy', allow_pickle=True)

    except:
        sel = adjusted_data[selected_features]
        X_train,y_train,X_test,y_test,populations,regions = split_for_training(sel)
        #Save
        np.save(outdir+'X_train.npy',X_train)
        np.save(outdir+'y_train.npy',y_train)
        np.save(outdir+'X_test.npy',X_test)
        np.save(outdir+'y_test.npy',y_test)
        np.save(outdir+'populations.npy',populations)
        np.save(outdir+'regions.npy',regions)



    return X_train,y_train,X_test,y_test,populations,regions

def split_for_training(sel):
    '''Split the data for training and testing
    '''
    X_train = [] #Inputs
    y_train = [] #Targets
    X_test = [] #Inputs
    y_test = [] #Targets
    countries = sel['Country_index'].unique()
    populations = []
    regions = []
    for ci in countries:
        country_data = sel[sel['Country_index']==ci]
        #Check regions
        country_regions = country_data['Region_index'].unique()
        for ri in country_regions:
            country_region_data = country_data[country_data['Region_index']==ri]
            #Select data 14 days before 0 cases
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
            #country_region_data['rescaled_cases']=country_region_data['rescaled_cases']/(population/100000)
            #country_region_data['cumulative_rescaled_cases']=country_region_data['cumulative_rescaled_cases']/(population/100000)

            #Loop through and get the first 21 days of data
            for di in range(len(country_region_data)-41):
                #Get change over the past 21 days
                xi = np.array(country_region_data.loc[di:di+20]).flatten()
                change_21 = xi[-country_region_data.shape[1]:][13]-xi[:country_region_data.shape[1]][13]
                #Add
                X_train.append(np.append(xi,[country_index,region_index,death_to_case_scale,case_death_delay,gross_net_income,population_density,change_21,pdi, idv, mas, uai, ltowvs, ivr, population]))
                y_train.append(np.array(country_region_data.loc[di+21:di+21+20]['smoothed_cases']))

            #Get the last 3 weeks as test
            X_test.append(X_train.pop())
            y_test.append(y_train.pop())
            #Save population
            populations.append(population)

    return np.array(X_train), np.array(y_train),np.array(X_test), np.array(y_test), np.array(populations), np.array(regions)

def fit_model(X_train,y_train,X_test,outdist):
    '''Fit the linear model
    '''
    try:
        #If the model has already been fitted
        corrs = np.load(outdir+'corrs.npy',allow_pickle=True)
        errors = np.load(outdir+'errors.npy',allow_pickle=True)
        stds = np.load(outdir+'stds.npy',allow_pickle=True)
        preds = np.load(outdir+'preds.npy',allow_pickle=True)
        coefs = np.load(outdir+'coefs.npy',allow_pickle=True)
    except:
        #Fit the model
        corrs = []
        errors = []
        stds = []
        preds = []
        coefs = []
        for i in range(y_train.shape[1]):
            reg = LinearRegression().fit(X_train, y_train[:,i])
            pred = reg.predict(X_test)
            #No negative predictions are allowed
            pred[pred<0]=0
            preds.append(pred)
            av_er = np.average(np.absolute(pred-y_test[:,i]))
            std = np.std(np.absolute(pred-y_test[:,i]))
            print('Error',av_er, 'Std',std)
            R,p = pearsonr(pred,y_test[:,i])
            #Save
            corrs.append(R)
            errors.append(av_er)
            stds.append(std)
            coefs.append(reg.coef_)
            #Plot
            plt.scatter(pred,y_test[:,i],s=1)
            plt.title(str(i))
            plt.xlabel('Predicted')
            plt.xlabel('True')
            plt.savefig(outdir+str(i)+'.png',format='png')
            plt.close()


        #Convert all to arrays
        corrs = np.array(corrs )
        errors = np.array(errors)
        stds = np.array(stds)
        preds = np.array(preds)
        coefs = np.array(coefs)
        #Save
        np.save(outdir+'corrs.npy',corrs)
        np.save(outdir+'errors.npy',errors)
        np.save(outdir+'stds.npy',stds)
        np.save(outdir+'preds.npy',preds)
        np.save(outdir+'coefs.npy',coefs)

    return corrs, errors, stds, preds, coefs

def evaluate_model(corrs, errors, stds, preds, coefs, y_test, outdir):
    '''Evaluate the fit model
    '''

    #Evaluate model
    results_file = open(outdir+'results.txt','w')
    total_regional_cum_error = []
    total_regional_mae = []
    total_regional_2week_mae = []
    total_regional_mae_per_100000 = []
    all_regional_corr = []
    #Evaluate the test cases
    for ri in range(len(regions)):
        #Plot
        region_error = np.cumsum(np.absolute(preds[:,ri]-y_test[ri,:]))[-1]
        total_regional_cum_error.append(region_error)
        total_regional_mae.append(np.average(np.absolute(preds[:,ri]-y_test[ri,:])))
        total_regional_mae_per_100000.append(np.average(np.absolute(preds[:,ri]-y_test[ri,:])/(populations[ri]/100000)))
        total_regional_2week_mae.append(np.average(np.absolute(preds[:,ri][:14]-y_test[ri,:][:14])))
        region_corr = pearsonr(preds[:,ri],y_test[ri,:])[0]
        all_regional_corr.append(region_corr)
        #fig, ax = plt.subplots(figsize=(6/2.54, 4/2.54))
        #plt.plot(range(1,22),preds[:,ri],label='pred',color='grey')
        #plt.plot(range(1,22),y_test[ri,:],label='true',color='g')
        #plt.title(regions[ri]+'\nPopulation:'+str(np.round(populations[ri]/1000000,1))+' millions\nCumulative error:'+str(np.round(region_error))+' PCC:'+str(np.round(region_corr,2)))
        #plt.xlabel('Day')
        #plt.ylabel('Cases')
        #ax.spines['top'].set_visible(False)
        #ax.spines['right'].set_visible(False)
        #fig.tight_layout()
        #plt.savefig(outdir+'regions/'+regions[ri]+'.png',format='png')
        #plt.close()
        results_file.write(regions[ri]+': '+str(region_corr)+'\n')
    #Convert to arrays
    total_regional_cum_error = np.array(total_regional_cum_error)
    total_regional_mae = np.array(total_regional_mae)
    all_regional_corr = np.array(all_regional_corr)
    #Calculate error
    results_file.write('Total 2week mae: '+str(np.sum(total_regional_2week_mae))+'\n')
    results_file.write('Total mae: '+str(np.sum(total_regional_mae))+'\n')
    results_file.write('Total mae per 100000: '+str(np.sum(total_regional_mae_per_100000))+'\n')
    results_file.write('Total cumulative error: '+str(np.sum(total_regional_cum_error))+'\n')
    #Evaluate all regions with at least 10 observed cases
    for t in [1,100,1000,10000]:
        results_file.write('Total normalized mae for regions with over '+str(t)+' observed cases: '+str(np.sum(total_regional_mae[np.where(np.sum(y_test,axis=1)>t)]/(np.sum(y_test[np.where(np.sum(y_test,axis=1)>t)],axis=1))))+'\n')

    #Set NaNs to 0
    all_regional_corr[np.isnan(all_regional_corr)]=0
    results_file.write('Average correlation: '+str(np.average(all_regional_corr)))
    results_file.close()

    #Look at coefs
    #The first are repeats 21 times, then single_features follow: [country_index,region_index,death_to_case_scale,case_death_delay,gross_net_income,population_density,population]
    #--> get the last features, then divide into 21 portions

    single_feature_names=['country_index','region_index','death_to_case_scale','case_death_delay','gross_net_income','population_density','Change in last 21 days','pdi', 'idv', 'mas', 'uai', 'ltowvs', 'ivr','population']
    #days pred,days behind - this goes from -21 to 1,features
    repeat_feature_names = ['C1_School closing', 'C2_Workplace closing', 'C3_Cancel public events', 'C4_Restrictions on gatherings', 'C5_Close public transport', 'C6_Stay at home requirements',
    'C7_Restrictions on internal movement', 'C8_International travel controls', 'H1_Public information campaigns', 'H2_Testing policy', 'H3_Contact tracing', 'H6_Facial Coverings',
    'rescaled_cases', 'cumulative_rescaled_cases', 'monthly_temperature', 'retail_and_recreation', 'grocery_and_pharmacy', 'parks','transit_stations', 'workplaces', 'residential']
    all_feature_names = single_feature_names+repeat_feature_names*21
    for i in range(coefs.shape[0]):
        fig,ax=plt.subplots(figsize=(18,6))
        plt.bar(range(coefs.shape[1]),coefs[i,:],)
        for j in range(coefs.shape[1]):
            plt.text(j,coefs[i,j],all_feature_names[j],fontsize=12)

        plt.title('Day '+str(i+1))
        plt.tight_layout()
        plt.savefig(outdir+'coefs_'+str(i+1)+'.png',format='png',dpi=300)
        plt.close()
    pdb.set_trace()
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
outdir = args.outdir[0]

#Use only data from July 1
adjusted_data = adjusted_data[adjusted_data['Date']>=start_date]


#Get data
X_train,y_train,X_test,y_test,populations,regions =  get_features(adjusted_data,outdir)
#Fit model
corrs, errors, stds, preds, coefs = fit_model(X_train,y_train,X_test,outdir)
#Evaluate model
evaluate_model(corrs, errors, stds, preds, coefs, y_test, outdir)
