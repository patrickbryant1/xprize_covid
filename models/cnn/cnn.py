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
from tensorflow.keras.callbacks import TensorBoard
from scipy.stats import pearsonr

import pdb
#Arguments for argparse module:
parser = argparse.ArgumentParser(description = '''A CNN regression model.''')

parser.add_argument('--adjusted_data', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to processed data file.')
parser.add_argument('--days_ahead', nargs=1, type= int,
                  default=sys.stdin, help = 'Number of days ahead to fit')
parser.add_argument('--start_date', nargs=1, type= str,
                  default=sys.stdin, help = 'Date to start from.')
parser.add_argument('--train_days', nargs=1, type= int,
                  default=sys.stdin, help = 'Days to include in fitting.')
parser.add_argument('--forecast_days', nargs=1, type= int,
                  default=sys.stdin, help = 'Days to forecast.')
#parser.add_argument('--param_combo', nargs=1, type= int,
                  #default=sys.stdin, help = 'Parameter combo.')
parser.add_argument('--outdir', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to output directory. Include /in end')

#######FUNCTIONS#######
def seed_everything(seed=2020):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def get_features(adjusted_data, train_days, forecast_days, outdir):
    '''Get the selected features
    '''

    #Get features
    try:
        X = np.load(outdir+'X.npy', allow_pickle=True)
        y = np.load(outdir+'y.npy', allow_pickle=True)
        populations = np.load(outdir+'populations.npy', allow_pickle=True)
        regions = np.load(outdir+'regions.npy', allow_pickle=True)


    except:
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
        X,y,populations,regions = split_for_training(sel,train_days,forecast_days)
        #Save
        np.save(outdir+'X.npy',X)
        np.save(outdir+'y.npy',y)
        np.save(outdir+'populations.npy',populations)
        np.save(outdir+'regions.npy',regions)



    return X,y,populations,regions



def split_for_training(sel, train_days, forecast_days):
    '''Split the data for training and testing
    '''
    X = [] #Inputs
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
            #Select data 14 days before 0 cases
            try:
                si = max(0,country_region_data[country_region_data['cumulative_rescaled_cases']>0].index[0]-14)
                country_region_data = country_region_data.loc[si:]
            except:
                print(len(country_region_data[country_region_data['cumulative_rescaled_cases']>0]),'cases for',country_region_data['CountryName'].unique()[0])
                continue

            country_region_data = country_region_data.reset_index()

            #Check if data
            if len(country_region_data)<train_days+forecast_days+1:
                print('Not enough data for',country_region_data['CountryName'].values[0])
                continue

            region_index = country_region_data.loc[0,'Region_index']
            if region_index!=0:
                regions.append(country_region_data.loc[0,'CountryName']+'_'+country_region_data.loc[0,'RegionName'])
            else:
                regions.append(country_region_data.loc[0,'CountryName'])

            population = country_region_data.loc[0,'population']
            country_region_data = country_region_data.drop(columns={'index','Country_index', 'Region_index','CountryName',
            'RegionName', 'death_to_case_scale', 'case_death_delay'})

            #Normalize the cases by 100'000 population
            #country_region_data['rescaled_cases']=country_region_data['rescaled_cases']/(population/100000)
            #country_region_data['cumulative_rescaled_cases']=country_region_data['cumulative_rescaled_cases']/(population/100000)
            country_region_data['smoothed_cases']=country_region_data['smoothed_cases']/(population/100000)
            country_region_data['cumulative_smoothed_cases']=country_region_data['cumulative_smoothed_cases']/(population/100000)
            #Get the data
            X.append(np.array(country_region_data))
            y.append(np.array(country_region_data['smoothed_cases']))

            #Save population
            populations.append(population)

    return np.array(X), np.array(y), np.array(populations), np.array(regions)

class DataGenerator(keras.utils.Sequence):
    '''Generates data for Keras'''
    def __init__(self, X_train_fold, y_train_fold, region_days, train_days,forecast_days, batch_size=1, shuffle=True):
        'Initialization'
        self.X_train_fold = X_train_fold
        self.y_train_fold = y_train_fold
        self.region_days = region_days
        self.train_days = train_days
        self.forecast_days = forecast_days
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(max(self.region_days)-self.train_days-self.forecast_days)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        batch_indices = self.indices+index
        # Generate data
        X_batch, y_batch = self.__data_generation(batch_indices)

        return X_batch, y_batch

    def on_epoch_end(self): #Will be done at epoch 0 also
        'Resets indices after each epoch'
        self.indices = np.repeat(self.train_days,self.X_train_fold.shape[0])



    def __data_generation(self, batch_indices):
        'Generates data containing batch_size samples'
        X_batch = []
        y_batch = []

        for i in range(self.X_train_fold.shape[0]):
            days_i = min(self.region_days[i]-self.forecast_days-self.train_days,batch_indices[i])
            X_batch.append(self.X_train_fold[i][days_i:days_i+self.train_days])
            y_batch.append(self.y_train_fold[i][days_i+self.train_days:days_i+self.train_days+self.forecast_days])

        return np.array(X_batch), np.array(y_batch)

#####LOSSES AND SCORES#####
def test(net, X_test,y_test,populations,regions):
    '''Test the net on the last 3 weeks of data
    '''

    test_preds=net.predict(np.array(X_test))
    R,p = pearsonr(test_preds[:,1],y_test)
    print('PCC:',R)
    order = np.argsort(y_test)
    plt.plot(y_test[order],test_preds[:,1][order],color='grey')
    plt.plot(y_test[order],y_test[order],color='k',linestyle='--')
    plt.fill_between(y_test[order],test_preds[:,0][order],test_preds[:,2][order],color='grey',alpha=0.5)
    plt.xlim([min(y_test),max(y_test)])
    plt.ylim([min(y_test),max(y_test)])
    plt.xlabel('True')
    plt.ylabel('Pred')
    plt.show()

#Custom loss
#============================#
def qloss(y_true, y_pred):
    # Pinball loss for multiple quantiles
    qs = [0.2, 0.50, 0.8]
    q = tf.constant(np.array([qs]), dtype=tf.float32)
    e = y_true - y_pred
    v = tf.maximum(q*e, (q-1)*e)
    return K.mean(v)



#####BUILD NET#####
def build_net():
    '''Build the net using Keras
    '''

    def resnet(x, num_res_blocks):
        """Builds a resnet with 1D convolutions of the defined depth.
        """


    x_in = keras.Input(shape= (None,32))
    #Initial convolution
    conv1 = L.Conv1D(filters = filters, kernel_size = kernel_size, dilation_rate = dilation_rate, padding ="same")(x_in)
    batch_out1 = L.BatchNormalization()(conv1)
    conv2 = L.Conv1D(filters = filters, kernel_size = kernel_size, dilation_rate = 7, padding ="same")(conv1)
    batch_out2 = L.BatchNormalization()(conv2)
    #Maxpool along sequence axis
    maxpool1 = L.GlobalMaxPooling1D()(batch_out2)

    preds = L.Dense(21, activation="relu", name="p2")(maxpool1)

    model = M.Model(x_in, preds, name="CNN")
    model.compile(loss='mae', optimizer=tf.keras.optimizers.Adagrad(lr=0.01),metrics=['mae'])
    return model


#####MAIN#####
args = parser.parse_args()
#Seed
seed_everything(42) #The answer it is
adjusted_data = pd.read_csv(args.adjusted_data[0],
                 parse_dates=['Date'],
                 encoding="ISO-8859-1",
                 dtype={"RegionName": str,
                        "RegionCode": str,
                        "Country_index":int,
                        "Region_index":int},
                 error_bad_lines=False)
adjusted_data = adjusted_data.fillna(0)
days_ahead = args.days_ahead[0]
start_date = args.start_date[0]
train_days = args.train_days[0]
forecast_days = args.forecast_days[0]
outdir = args.outdir[0]
#Use only data from start date
adjusted_data = adjusted_data[adjusted_data['Date']>=start_date]
#Get features
X,y,populations,regions  = get_features(adjusted_data,train_days,forecast_days,outdir)
#Get number of days in X
num_days = []
for cr in range(len(X)):
    num_days.append(X[cr].shape[0])
num_days = np.array(num_days)

#Get net parameters
BATCH_SIZE=int(len(X)*0.8)
EPOCHS=100
dilation_rate = 3
kernel_size = 5
filters = 32
#Make net

net = build_net()
print(net.summary())
#KFOLD
NFOLD = 5
kf = KFold(n_splits=NFOLD)
fold=0

#Save errors
errors = []
corrs = []
for tr_idx, val_idx in kf.split(X):
    fold+=1
    tensorboard = TensorBoard(log_dir=outdir+'fold'+str(fold))
    print("FOLD", fold)
    net = build_net()
    #Data generation
    training_generator = DataGenerator(X[tr_idx], y[tr_idx],num_days[tr_idx],train_days,forecast_days, BATCH_SIZE)
    valid_generator = DataGenerator(X[val_idx], y[val_idx],num_days[val_idx],train_days,forecast_days, BATCH_SIZE)

    net.fit(training_generator,
            validation_data=valid_generator,
            epochs=EPOCHS,
            callbacks = [tensorboard]
            )

    preds = net.predict(X[val_idx])
    preds[preds<0]=0
    errors.append(np.average(np.absolute(preds[:,1]-y[val_idx])))
    corrs.append(pearsonr(preds[:,1],y[val_idx])[0])
print(np.average(errors))
np.average(corrs)
pdb.set_trace()
