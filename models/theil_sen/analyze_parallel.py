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
from sklearn.linear_model import TheilSenRegressor
import numpy as np


import pdb
#Arguments for argparse module:
parser = argparse.ArgumentParser(description = '''Analyze Theil-Sen model.''')

parser.add_argument('--indir', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to processed data file.')
parser.add_argument('--outdir', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to output directory. Include /in end')

def predict(X_train,y_train,coefs,intercept,day):
    '''Predict to analyze error
    '''

    train_pred = np.dot(X_train,coefs)+intercept
    #Negative not allowed
    train_pred[train_pred<0]=0
    train_errors = np.absolute(train_pred-y_train)
    av_train_error = np.average(train_errors)
    std_train_error = np.std(train_errors)
    R,p = pearsonr(train_pred,y_train)
    fig,ax = plt.subplots(figsize=(6/2.54,6/2.54))
    plt.scatter(train_pred,y_train,s=1)
    plt.title('Av.er:'+str(np.round(av_train_error,3))+'|std:'+str(np.round(std_train_error,3))+'|PCC:'+str(np.round(R,2)))
    fig.tight_layout()
    plt.savefig(outdir+'train_true'+str(day)+'.png',format='png')
    plt.close()
    return av_train_error, std_train_error, R


def evaluate(preds,coefs,intercepts,X_train,y_train,y_test,outdir,regions,populations):
    '''Evaluate the model
    '''

    #2.Evaluate the test cases
    #Evaluate model
    results_file = open(outdir+'results.txt','w')
    total_regional_cum_error = []
    total_regional_mae = []
    total_regional_mae_per_100000 = []
    total_regional_2week_mae = []
    all_regional_corr = []
    #Evaluate the test cases
    for ri in range(len(regions)):
        #Plot
        region_error = np.cumsum(np.absolute(preds[:,ri]-y_test[ri,:]))[-1]
        total_regional_cum_error.append(region_error)
        total_regional_mae.append(np.average(np.absolute(preds[:,ri]-y_test[ri,:])*(populations[ri]/100000)))
        total_regional_mae_per_100000.append(np.average(np.absolute(preds[:,ri]-y_test[ri,:])))
        total_regional_2week_mae.append(np.average(np.absolute(preds[:,ri][:14]-y_test[ri,:][:14])*(populations[ri]/100000)))
        region_corr = pearsonr(preds[:,ri],y_test[ri,:])[0]
        all_regional_corr.append(region_corr)
        fig, ax = plt.subplots(figsize=(6/2.54, 4/2.54))
        # plt.plot(range(1,22),preds[:,ri],label='pred',color='grey')
        # plt.plot(range(1,22),y_test[ri,:],label='true',color='g')
        # plt.title(regions[ri]+'\nPopulation:'+str(np.round(populations[ri]/1000000,1))+' millions\nCumulative error:'+str(np.round(region_error))+' PCC:'+str(np.round(region_corr,2)))
        # plt.xlabel('Day')
        # plt.ylabel('Cases')
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        # fig.tight_layout()
        # plt.savefig(outdir+'regions/'+regions[ri]+'.png',format='png')
        # plt.close()
        results_file.write(regions[ri]+': '+str(region_corr)+'\n')

    #Convert to arrays
    total_regional_cum_error = np.array(total_regional_cum_error)
    total_regional_mae = np.array(total_regional_mae)
    all_regional_corr = np.array(all_regional_corr)
    #Calculate error
    results_file.write('Total 2week mae: '+str(np.sum(total_regional_2week_mae))+'\n')
    results_file.write('Total mae: '+str(np.sum(total_regional_mae))+'\n')
    results_file.write('Total mae per 100000: '+str(np.sum(total_regional_mae_per_100000))+'\n')
    results_file.write('Average mae per 100000: '+str(np.average(total_regional_mae_per_100000))+'\n')
    results_file.write('Total cumulative error: '+str(np.sum(total_regional_cum_error))+'\n')
    #Evaluate all regions with at least 10 observed cases
    for t in [1,100,1000,10000]:
        results_file.write('Total normalized mae for regions with over '+str(t)+' observed cases: '+str(np.sum(total_regional_mae[np.where(np.sum(y_test,axis=1)>t)]/(np.sum(y_test[np.where(np.sum(y_test,axis=1)>t)],axis=1))))+'\n')

    #Set NaNs to 0
    all_regional_corr[np.isnan(all_regional_corr)]=0
    results_file.write('Average correlation: '+str(np.average(all_regional_corr)))


    #Look at coefs
    #The first are repeats 21 times, then single_features follow: [country_index,region_index,death_to_case_scale,case_death_delay,gross_net_income,population_density,population]
    #--> get the last features, then divide into 21 portions
    single_feature_names=['country_index','region_index','death_to_case_scale','case_death_delay','gross_net_income','population_density','Change in last 21 days','pdi', 'idv', 'mas', 'uai', 'ltowvs', 'ivr','population']
    #days pred,days behind - this goes from -21 to 1,features
    repeat_feature_names = ['C1_School closing', 'C2_Workplace closing', 'C3_Cancel public events', 'C4_Restrictions on gatherings', 'C5_Close public transport', 'C6_Stay at home requirements',
    'C7_Restrictions on internal movement', 'C8_International travel controls', 'H1_Public information campaigns', 'H2_Testing policy', 'H3_Contact tracing', 'H6_Facial Coverings',
    'rescaled_cases', 'cumulative_rescaled_cases', 'monthly_temperature', 'retail_and_recreation', 'grocery_and_pharmacy', 'parks','transit_stations', 'workplaces', 'residential']
    all_feature_names = single_feature_names+repeat_feature_names*21
    # for i in range(coefs.shape[0]):
    #     fig,ax=plt.subplots(figsize=(18,6))
    #     plt.bar(range(coefs.shape[1]),coefs[i,:],)
    #     #for j in range(coefs.shape[1]):
    #     #    plt.text(j,coefs[i,j],all_feature_names[j],fontsize=12)
    #     plt.title('Day '+str(i+1))
    #     plt.tight_layout()
    #     plt.savefig(outdir+'coefs_'+str(i+1)+'.png',format='png',dpi=300)
    #     plt.close()

    #Analyze error per day ahead
    av_err_per_day = []
    std_per_day = []
    corr_per_day = []

    for da in range(preds.shape[0]):
        av_train_error, std_train_error, R = predict(X_train,y_train[:,da],coefs[da,:],intercepts[da],da+1)
        av_err_per_day.append(av_train_error)
        std_per_day.append(std_train_error)
        corr_per_day.append(R)
    #Convert to arrays
    av_err_per_day = np.array(av_err_per_day)
    std_per_day = np.array(std_per_day)
    corr_per_day = np.array(corr_per_day)
    #Plot average error per day with std
    fig, ax = plt.subplots(figsize=(9/2.54, 6/2.54))
    plt.plot(range(1,len(av_err_per_day)+1),av_err_per_day,color='b')
    #plt.fill_between(range(1,len(av_err_per_day)+1),av_err_per_day-std_per_day,av_err_per_day+std_per_day,color='b',alpha=0.5)
    plt.title('Average error')
    plt.xlabel('Days in the future')
    plt.ylabel('Error per 100000')
    plt.tight_layout()
    plt.savefig(outdir+'lr_av_error.png',format='png', dpi=300)
    plt.close()
    pdb.set_trace()
    #Plot correlation
    fig, ax = plt.subplots(figsize=(9/2.54, 6/2.54))
    plt.plot(range(1,len(corr_per_day)+1),corr_per_day ,color='b')
    plt.title('Pearson R')
    plt.xlabel('Days in the future')
    plt.ylabel('PCC')
    plt.tight_layout()
    plt.savefig(outdir+'PCC.png',format='png', dpi=300)
    plt.close()

    results_file.close()
    return None

#####MAIN#####
#Set font size
matplotlib.rcParams.update({'font.size': 7})
args = parser.parse_args()
indir = args.indir[0]
outdir = args.outdir[0]

#Get features
X_train = np.load(indir+'X_train.npy', allow_pickle=True)
y_train = np.load(indir+'y_train.npy', allow_pickle=True)
X_test = np.load(indir+'X_test.npy', allow_pickle=True)
y_test = np.load(indir+'y_test.npy', allow_pickle=True)
populations = np.load(indir+'populations.npy', allow_pickle=True)
regions = np.load(indir+'regions.npy', allow_pickle=True)





preds = []
coefs = []
intercepts = []
for day in range(1,22):
    preds.append(np.load(indir+'preds/preds'+str(day)+'.npy',allow_pickle=True))
    coefs.append(np.load(indir+'coefficients/coefficients'+str(day)+'.npy',allow_pickle=True))
    intercepts.append(np.load(indir+'intercepts/intercept'+str(day)+'.npy',allow_pickle=True))
preds = np.array(preds)
coefs = np.array(coefs)
intercepts = np.array(intercepts)

#Evaluate fit
evaluate(preds,coefs,intercepts,X_train,y_train,y_test,outdir,regions,populations)
