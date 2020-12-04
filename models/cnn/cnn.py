#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import sys
import os
import numpy as np
import random
import pandas as pd

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
parser.add_argument('--start_date', nargs=1, type= str,
                  default=sys.stdin, help = 'Date to start from.')
parser.add_argument('--train_days', nargs=1, type= int,
                  default=sys.stdin, help = 'Days to include in fitting.')
parser.add_argument('--forecast_days', nargs=1, type= int,
                  default=sys.stdin, help = 'Days to forecast.')
parser.add_argument('--param_combo', nargs=1, type= str,
                  default=sys.stdin, help = 'Parameter combo.')
parser.add_argument('--datadir', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to data directory. Include /in end')
parser.add_argument('--outdir', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to output directory. Include /in end')

#######FUNCTIONS#######
def seed_everything(seed=2020):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def read_net_params(params_file):
    '''Read and return net parameters
    '''
    net_params = {} #Save information for net

    with open(params_file) as file:
        for line in file:
            line = line.rstrip() #Remove newlines
            line = line.split("=") #Split on "="

            net_params[line[0]] = line[1]


    return net_params

def normalize_data(sel):
    '''Normalize and transform data
    '''

    # to_log = ['smoothed_cases','cumulative_smoothed_cases','rescaled_cases','cumulative_rescaled_cases','population_density', 'population']
    # for var in to_log:
    #     sel[var] = np.log10(sel[var]+0.001)

    #GNI: group into 3: 0-20k,20-40k,40k+
    index1 = sel[sel['gross_net_income']<20000].index
    above = sel[sel['gross_net_income']>20000]
    index2 = above[above['gross_net_income']<40000].index
    index3 = sel[sel['gross_net_income']>40000].index
    sel.at[index1,'gross_net_income']=0
    sel.at[index2,'gross_net_income']=1
    sel.at[index3,'gross_net_income']=2

    return sel

def get_features(adjusted_data, train_days, forecast_days, datadir):
    '''Get the selected features
    '''

    #Get features
    try:
        X = np.load(datadir+'X.npy', allow_pickle=True)
        y = np.load(datadir+'y.npy', allow_pickle=True)
        populations = np.load(datadir+'populations.npy', allow_pickle=True)
        regions = np.load(datadir+'regions.npy', allow_pickle=True)


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
        #Normalize
        sel = normalize_data(sel)
        X,y,populations,regions = split_for_training(sel,train_days,forecast_days)
        #Save
        np.save(datadir+'X.npy',X)
        np.save(datadir+'y.npy',y)
        np.save(datadir+'populations.npy',populations)
        np.save(datadir+'regions.npy',regions)



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
            country_region_data['rescaled_cases']=country_region_data['rescaled_cases']/(population/100000)
            country_region_data['cumulative_rescaled_cases']=country_region_data['cumulative_rescaled_cases']/(population/100000)
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
        self.cum_region_days = np.cumsum(self.region_days-self.train_days-self.forecast_days)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.sum(self.region_days-self.train_days-self.forecast_days))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        batch_index = self.indices+index
        region_index = np.argwhere(self.cum_region_days>batch_index)[0][0]
        #Increase the right region index
        self.region_indices[region_index]+=1
        # Generate data
        X_batch, y_batch = self.__data_generation(region_index,self.region_indices[region_index])

        return X_batch, y_batch

    def on_epoch_end(self): #Will be done at epoch 0 also
        'Resets indices after each epoch'
        self.indices = 0
        self.region_indices = np.repeat(self.train_days-1,self.X_train_fold.shape[0])

    def __data_generation(self, region_index,batch_end_day):
        'Generates data containing batch_size samples'
        #Get the region
        X_batch = []
        y_batch = []

        X_batch.append(self.X_train_fold[region_index][:batch_end_day])
        y_batch.append(self.y_train_fold[region_index][batch_end_day:batch_end_day+self.forecast_days])

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


#####BUILD NET#####
def build_net():
    '''Build the net using Keras
    '''


    x_in = keras.Input(shape= (None,32))
    #Convolutions
    def get_conv_net(x,num_convolutional_layers):
        for n in range(num_convolutional_layers):
            x = L.Conv1D(filters = filters, kernel_size = kernel_size, dilation_rate = dilation_rate, padding ="same")(x)
            x = L.BatchNormalization()(x)
            x = L.Activation('relu')(x)

        return x

    x = get_conv_net(x_in,num_convolutional_layers)
    #Maxpool along sequence axis
    maxpool1 = L.GlobalMaxPooling1D()(x)

    preds = L.Dense(21, activation="relu", name="p2")(maxpool1)

    model = M.Model(x_in, preds, name="CNN")
    model.compile(loss='mae', optimizer=tf.keras.optimizers.Adagrad(lr=lr))
    return model


#####MAIN#####
args = parser.parse_args()
#Seed
#seed_everything(0) #The answer it is
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
datadir = args.datadir[0]
outdir = args.outdir[0]
#Use only data from start date
adjusted_data = adjusted_data[adjusted_data['Date']>=start_date]
#Get features
X,y,populations,regions  = get_features(adjusted_data,train_days,forecast_days, datadir)
#Get number of days in X
num_days = []
for cr in range(len(X)):
    num_days.append(X[cr].shape[0])
num_days = np.array(num_days)

#Get net parameters
net_params = read_net_params(args.param_combo[0])
BATCH_SIZE=1
EPOCHS=1
filters = int(net_params['filters']) #32
dilation_rate = int(net_params['dilation_rate'])#3
kernel_size = int(net_params['kernel_size']) #5
lr = float(net_params['lr']) #0.01
num_convolutional_layers = int(net_params['num_convolutional_layers'])
#Make net
net = build_net()
print(net.summary())
#KFOLD
NFOLD = 5
kf = KFold(n_splits=NFOLD,shuffle=True, random_state=0)
fold=0

#Save errors
train_errors = []
valid_errors = []
corrs = []
for tr_idx, val_idx in kf.split(X):
    fold+=1
    #tensorboard = TensorBoard(log_dir=outdir+'fold'+str(fold))
    print("FOLD", fold)
    net = build_net()
    #Data generation
    training_generator = DataGenerator(X[tr_idx], y[tr_idx],num_days[tr_idx],train_days,forecast_days, BATCH_SIZE)
    valid_generator = DataGenerator(X[val_idx], y[val_idx],num_days[val_idx],train_days,forecast_days, BATCH_SIZE)

    history = net.fit(training_generator,
            validation_data=valid_generator,
            epochs=EPOCHS
            )
    #Save loss and accuracy
    train_errors.append(np.array(history.history['loss']))
    valid_errors.append(np.array(history.history['val_loss']))

np.save(outdir+'train_errors.npy', np.array(train_errors))
np.save(outdir+'valid_errors.npy', np.array(valid_errors))
