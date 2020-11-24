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
    populations = []
    for ci in countries:
        country_data = sel[sel['Country_index']==ci]
        #Check regions
        country_regions = country_data['Region_index'].unique()
        for ri in country_regions:
            country_region_data = country_data[country_data['Region_index']==ri]
            country_region_data = country_region_data[country_region_data['cumulative_rescaled_cases']>0]
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
            population = country_region_data.loc[0,'population']
            country_region_data = country_region_data.drop(columns={'index','Country_index', 'Region_index','death_to_case_scale', 'case_death_delay', 'gross_net_income','population_density','pdi', 'idv', 'mas', 'uai', 'ltowvs', 'ivr','population'})

            #Normalize the cases by 100'000 population
            country_region_data['rescaled_cases']=country_region_data['rescaled_cases']#/(population/100000)
            country_region_data['cumulative_rescaled_cases']=country_region_data['cumulative_rescaled_cases']#/(population/100000)

            #Loop through and get the first 21 days of data
            for di in range(len(country_region_data)-41):
                #Get change over the past 21 days
                xi = np.array(country_region_data.loc[di:di+20]).flatten()
                change_21 = xi[-country_region_data.shape[1]:][13]-xi[:country_region_data.shape[1]][13]
                #Add
                X_train.append(np.append(xi,[country_index,region_index,death_to_case_scale,case_death_delay,gross_net_income,population_density,change_21,pdi, idv, mas, uai, ltowvs, ivr, population]))
                y_train.append(np.array(country_region_data.loc[di+21:di+21+20]['rescaled_cases']))

            #Get the last 3 weeks as test
            X_test.append(X_train.pop())
            y_test.append(y_train.pop())
            #Save population
            populations.append(population)

    return np.array(X_train), np.array(y_train),np.array(X_test), np.array(y_test), np.array(populations)


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
    populations = np.load(outdir+'populations.npy', allow_pickle=True)
except:
    sel=get_features(adjusted_data)
    X_train,y_train,X_test,y_test,populations = split_for_training(sel)
    #Save
    np.save(outdir+'X_train.npy',X_train)
    np.save(outdir+'y_train.npy',y_train)
    np.save(outdir+'X_test.npy',X_test)
    np.save(outdir+'y_test.npy',y_test)
    np.save(outdir+'populations.npy',populations)

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
    av_er = np.average(np.absolute(pred-y_test[:,i])/populations)
    std = np.std(np.absolute(pred-y_test[:,i])/populations)
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



preds = np.array(preds)
#Plot a test case
plt.plot(range(1,22),preds[:,0],label='pred')
plt.plot(range(1,22),y_test[0,:],label='true')
plt.savefig(outdir+'pred_and_true_sel.png',format='png')
plt.close()

#Look at coefs
coefs = np.array(coefs)

#The first are repeats 21 times, then single_features follow: [country_index,region_index,death_to_case_scale,case_death_delay,gross_net_income,population_density,population]
#--> get the last features, then divide into 21 portions

single_feature_names=['country_index','region_index','death_to_case_scale','case_death_delay','gross_net_income','population_density','Change in last 21 days','pdi', 'idv', 'mas', 'uai', 'ltowvs', 'ivr','population']
single_features=coefs[:,-len(single_feature_names):]
plt.imshow(single_features)
plt.yticks(range(21),labels=range(1,22))
plt.xticks(range(len(single_feature_names)),labels=single_feature_names,rotation='vertical')
plt.colorbar()
plt.tight_layout()
plt.savefig(outdir+'single_features.png',format='png',dpi=300)
plt.close()
remainder=coefs[:,:-8]
remainder=np.reshape(remainder,(21,21,-1)) #days pred,days behind - this goes from -21 to 1,features
remainder_names = ['C1_School closing', 'C2_Workplace closing', 'C3_Cancel public events', 'C4_Restrictions on gatherings', 'C5_Close public transport', 'C6_Stay at home requirements',
'C7_Restrictions on internal movement', 'C8_International travel controls', 'H1_Public information campaigns', 'H2_Testing policy', 'H3_Contact tracing', 'H6_Facial Coverings',
'rescaled_cases', 'cumulative_rescaled_cases', 'monthly_temperature', 'retail_and_recreation', 'grocery_and_pharmacy', 'parks','transit_stations', 'workplaces', 'residential']

for i in range(remainder.shape[2]):
    plt.imshow(remainder[:,:,i])
    #The first axis will end up horizontal, the second vertical
    plt.xlabel('Future day')
    plt.ylabel('Previous day')

    plt.xticks(range(21),labels=range(1,22))
    plt.yticks(range(21),labels=range(-21,0,1))
    plt.colorbar()
    plt.title(remainder_names[i])
    plt.tight_layout()
    plt.savefig(outdir+'feature_'+str(i)+'.png',format='png',dpi=300)
    plt.close()

#Plot average error per day with std
errors = np.array(errors)
std = np.array(stds)
plt.plot(range(1,22),errors,color='b')
plt.fill_between(range(1,22),errors-stds,errors+stds,color='b',alpha=0.5)
plt.title('Average error with std')
plt.xlabel('Days in the future')
plt.ylabel('Error per 100000')
plt.savefig(outdir+'lr_av_error.png',format='png')
plt.close()

#Plot correlation
corrs = np.array(corrs )
plt.plot(range(1,22),corrs ,color='b')
plt.title('Pearson R')
plt.xlabel('Days in the future')
plt.ylabel('PCC')
plt.savefig(outdir+'PCC.png',format='png')
plt.close()
