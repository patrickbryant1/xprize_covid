#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import sys
import os
import numpy as np
import random
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import tensorflow.keras.layers as L
import tensorflow.keras.models as M

from scipy.stats import pearsonr

import pdb
#Arguments for argparse module:
parser = argparse.ArgumentParser(description = '''A LSTM model.''')

parser.add_argument('--adjusted_data', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to processed data file.')
#parser.add_argument('--param_combo', nargs=1, type= int, default=sys.stdin, help = 'Parameter combo.')
parser.add_argument('--outdir', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to output directory. Include /in end')

#######FUNCTIONS#######
def seed_everything(seed=2020):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

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
                        'CountryName',
                        'RegionName',
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
    regions = []
    for ci in countries:
        country_data = sel[sel['Country_index']==ci]
        #Check regions
        country_regions = country_data['Region_index'].unique()
        for ri in country_regions:
            try:
                si = max(0,country_region_data[country_region_data['cumulative_rescaled_cases']>0].index[0]-14)
                country_region_data = country_region_data.loc[si:]
            except:
                print(len(country_region_data[country_region_data['cumulative_rescaled_cases']>0]),'cases for',country_region_data['CountryName'].unique()[0])
                continue
            population = country_region_data.loc[0,'population']
            country_region_data = country_region_data.reset_index()

            #Check if data
            if len(country_region_data)<1:
                continue


            if region_index!=0:
                regions.append(country_region_data.loc[0,'CountryName']+'_'+country_region_data.loc[0,'RegionName'])
            else:
                regions.append(country_region_data.loc[0,'CountryName'])
            country_region_data = country_region_data.drop(columns={'index','CountryName','RegionName',})

            #Normalize the cases by 100'000 population?
            #country_region_data['rescaled_cases']=country_region_data['rescaled_cases']#/(population/100000)
            #country_region_data['cumulative_rescaled_cases']=country_region_data['cumulative_rescaled_cases']#/(population/100000)

            #Loop through and get the first 21 days of data
            for di in range(len(country_region_data)-41):
                #Add
                X_train.append(country_region_data.loc[di:di+20])
                y_train.append(np.array(country_region_data.loc[di+21:di+21+20]['rescaled_cases']))
                pdb.set_trace()
            #Get the last 3 weeks as test
            X_test.append(X_train.pop())
            y_test.append(y_train.pop())
            #Save population
            populations.append(population)
            pdb.set_trace()
    return np.array(X_train), np.array(y_train),np.array(X_test), np.array(y_test), np.array(populations), np.array(regions)


#####LOSSES AND SCORES#####
def score(y_true, y_pred):
    tf.dtypes.cast(y_true, tf.float32)
    tf.dtypes.cast(y_pred, tf.float32)
    sigma = y_pred[:, 2] - y_pred[:, 0]
    #Make sure all y-pred are non-negative
    delta = tf.abs(y_true - y_pred[:, 1])

    return K.mean(delta)
#============================#
def qloss(y_true, y_pred):
    # Pinball loss for multiple quantiles
    qs = [0.05, 0.50, 0.95]
    q = tf.constant(np.array([qs]), dtype=tf.float32)
    e = y_true - y_pred
    v = tf.maximum(q*e, (q-1)*e)
    return K.mean(v)
#=============================#
def mloss(_lambda):
    def loss(y_true, y_pred):
        return _lambda * qloss(y_true, y_pred) + (1 - _lambda)*score(y_true, y_pred)
    return loss

#####BUILD NET#####
def build_net(n1,n2,input_dim):
    '''Build the net using Keras
    '''
    z = L.Input((input_dim,), name="Patient")
    x1 = L.LSTM(n1, activation="relu", name="d1")(z)
    x2 = L.LSTM(n2, activation="relu", name="d2")(x1)
    p1 = L.Dense(3, activation="linear", name="p1")(x2)
    p2 = L.Dense(3, activation="relu", name="p2")(x2)
    preds = L.Lambda(lambda x: x[0] + tf.cumsum(x[1], axis=1),
                     name="preds")([p1, p2])
    #Ensure non-negative values
    preds = K.abs(preds)
    model = M.Model(z, preds, name="CNN")
    model.compile(loss=mloss(0.8), optimizer=tf.keras.optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.01, amsgrad=False), metrics=[score])
    return model

def test(net, X_test,y_test,populations,regions):
    '''Test the net on the last 3 weeks of data
    '''
    for xi in range(X_test.shape[0]):
        preds_i = np.zeros((21,3))
        for day in range(21):
            preds_i[day]=net.predict(np.array([np.append(X_test[xi],day)]))

        fig,ax = plt.subplots(figsize=(6/2.54,4/2.54))
        plt.plot(np.arange(1,22),y_test[xi],color='g')
        plt.plot(np.arange(1,22),preds_i[:,1],color='grey')
        plt.fill_between(np.arange(1,22),preds_i[:,0],preds_i[:,2],color='grey',alpha=0.5)
        plt.title(regions[xi]+'\n'+str(np.round(populations[xi]/1000000,2))+' millions')
        plt.tight_layout()
        plt.savefig(outdir+regions[xi]+'.png',dpi=300,format='png')
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
outdir = args.outdir[0]

#Get features
try:
    X_train = np.load(outdir+'X_train.npy', allow_pickle=True)
    y_train = np.load(outdir+'y_train.npy', allow_pickle=True)
    X_test = np.load(outdir+'X_test.npy', allow_pickle=True)
    y_test = np.load(outdir+'y_test.npy', allow_pickle=True)
    populations = np.load(outdir+'populations.npy', allow_pickle=True)
    regions = np.load(outdir+'regions.npy', allow_pickle=True)
except:
    sel=get_features(adjusted_data)
    X_train,y_train,X_test,y_test,populations,regions = split_for_training(sel)
    #Save
    np.save(outdir+'X_train.npy',X_train)
    np.save(outdir+'y_train.npy',y_train)
    np.save(outdir+'X_test.npy',X_test)
    np.save(outdir+'y_test.npy',y_test)
    np.save(outdir+'populations.npy',populations)
    np.save(outdir+'regions.npy',regions)

#Seed
seed_everything(42) #The answer it is

#Get net parameters
BATCH_SIZE=256
EPOCHS=100
n1=100 #Nodes layer 1
n2=100 #Nodes layer 2
#Make net
net = build_net(n1,n2,X_train.shape[1]+1)
print(net.summary())
#KFOLD
NFOLD = 5
kf = KFold(n_splits=NFOLD)
fold=0

for tr_idx, val_idx in kf.split(X_train):
    fold+=1
    print("FOLD", fold)
    net = build_net(n1,n2,X_train.shape[1]+1)
    #Data generation
    training_generator = DataGenerator(X_train[tr_idx], y_train[tr_idx], BATCH_SIZE)
    valid_generator = DataGenerator(X_train[val_idx], y_train[val_idx], BATCH_SIZE)

    net.fit(training_generator,
            validation_data=valid_generator,
            epochs=EPOCHS
            )

    #Test the net
    test(net, X_test,y_test,populations,regions)
    pdb.set_trace()
pdb.set_trace()
