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

import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as L
import tensorflow.keras.models as M

from scipy.stats import pearsonr

import pdb
#Arguments for argparse module:
parser = argparse.ArgumentParser(description = '''A multiple Quantile regression model.''')

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

#####LOSSES AND SCORES#####
def score(y_true, y_pred):
    tf.dtypes.cast(y_true, tf.float32)
    tf.dtypes.cast(y_pred, tf.float32)
    sigma = y_pred[:, 2] - y_pred[:, 0]
    #Make sure all y-pred are non-negative
    delta = tf.abs(y_true[:, 0] - y_pred[:, 1])

    return K.mean(delta)
#============================#
def qloss(y_true, y_pred):
    # Pinball loss for multiple quantiles
    qs = [0.2, 0.50, 0.8]
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
    x = L.Dense(n1, activation="relu", name="d1")(z)
    x = L.Dense(n2, activation="relu", name="d2")(x)
    p1 = L.Dense(3, activation="linear", name="p1")(x)
    p2 = L.Dense(3, activation="relu", name="p2")(x)
    preds = L.Lambda(lambda x: x[0] + tf.cumsum(x[1], axis=1),
                     name="preds")([p1, p2])

    model = M.Model(z, preds, name="CNN")
    model.compile(loss=mloss(0.8), optimizer=tf.keras.optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.01, amsgrad=False), metrics=[score])
    return model

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

#Seed
seed_everything(42) #The answer it is

#Make net

net = build_net(100,10,X_train.shape[1])
print(net.summary())
pdb.set_trace()
