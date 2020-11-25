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
            country_region_data = country_data[country_data['Region_index']==ri]
            try:
                si = max(0,country_region_data[country_region_data['cumulative_rescaled_cases']>0].index[0]-14)
                country_region_data = country_region_data.loc[si:]
            except:
                print(len(country_region_data[country_region_data['cumulative_rescaled_cases']>0]),'cases for',country_region_data['CountryName'].unique()[0])
                continue

            country_region_data = country_region_data.reset_index()
            population = country_region_data.loc[0,'population']

            #Check if data
            if len(country_region_data)<1:
                continue


            if ri!=0:
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
                X_train.append(np.array(country_region_data.loc[di:di+20]))
                y_train.append(np.array(country_region_data.loc[di+21:di+21+20]['rescaled_cases']))

            #Get the last 3 weeks as test
            X_test.append(X_train.pop())
            y_test.append(y_train.pop())
            #Save population
            populations.append(population)

    return np.array(X_train), np.array(y_train),np.array(X_test), np.array(y_test), np.array(populations), np.array(regions)

class DataGenerator(keras.utils.Sequence):
    '''Generates data for Keras'''
    def __init__(self, X_train_fold, y_train_fold, batch_size=1, shuffle=True):
        'Initialization'
        self.X_train_fold = X_train_fold
        self.y_train_fold = y_train_fold
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.X_train_fold) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        batch_indices = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        #domain_index = np.take(range((len(self.X_train_fold))),indexes)

        # Generate data
        X_batch, y_batch = self.__data_generation(batch_indices)

        return X_batch, y_batch

    def on_epoch_end(self): #Will be done at epoch 0 also
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.X_train_fold))
        np.random.shuffle(self.indexes)


    def __data_generation(self, batch_indices):
        'Generates data containing batch_size samples'

        return self.X_train_fold[batch_indices],self.y_train_fold[batch_indices]


#####LOSSES AND SCORES#####
def bin_loss(y_true, y_pred):

    g_loss = tf.keras.losses.mean_absolute_error(y_true, y_pred) #general, compare difference
    kl_loss = tf.keras.losses.kullback_leibler_divergence(y_true, y_pred) #better than comparing to gaussian
    sum_kl_loss = K.sum(kl_loss, axis =0)
    sum_g_loss = K.sum(g_loss, axis =0)
    #sum_g_loss = sum_g_loss*10 #This is basically a loss penalty
    loss = sum_g_loss+sum_kl_loss
    return loss

#####BUILD NET#####
def build_net(n1,n2,input_dim):
    '''Build the net using Keras
    '''
    z = L.Input(input_dim, name="Patient")
    x1 = L.LSTM(n1, activation="relu", name="d1", return_sequences=True)(z)
    x2 = L.LSTM(n2, activation="relu", name="d2", return_sequences=True)(x1)

    preds = L.Dense(1, activation="relu", name="p2")(x2)
    model = M.Model(z, preds, name="CNN")
    model.compile(loss=bin_loss, optimizer=tf.keras.optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.01, amsgrad=False), metrics=['mae','kullback_leibler_divergence'])
    return model

def test(net, X_test,y_test,populations,regions):
    '''Test the net on the last 3 weeks of data
    '''
    for xi in range(X_test.shape[0]):
        preds_i = np.zeros((21,3))
        for day in range(21):
            preds_i[day]=net.predict(np.array([np.append(X_test[xi],day)]))

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

net = build_net(n1,n2,X_train.shape[1:])
print(net.summary())
#KFOLD
NFOLD = 5
kf = KFold(n_splits=NFOLD)
fold=0

for tr_idx, val_idx in kf.split(X_train):
    fold+=1
    print("FOLD", fold)
    net = build_net(n1,n2,X_train.shape[1:])
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
