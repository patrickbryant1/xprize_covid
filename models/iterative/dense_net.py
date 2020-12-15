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
parser = argparse.ArgumentParser(description = '''A dense regression model.''')

parser.add_argument('--adjusted_data', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to processed data file.')
parser.add_argument('--start_date', nargs=1, type= str,
                  default=sys.stdin, help = 'Date to start from.')
parser.add_argument('--train_days', nargs=1, type= int,
                  default=sys.stdin, help = 'Days to include in fitting.')
parser.add_argument('--forecast_days', nargs=1, type= int,
                  default=sys.stdin, help = 'Days to forecast.')
#parser.add_argument('--param_combo', nargs=1, type= int,
                  #default=sys.stdin, help = 'Parameter combo.')
parser.add_argument('--threshold', nargs=1, type= float,
                  default=sys.stdin, help = 'Threshold.')
parser.add_argument('--outdir', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to output directory. Include /in end')

#######FUNCTIONS#######
def seed_everything(seed=2020):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

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
                        'retail_and_recreation',
                        'grocery_and_pharmacy',
                        'parks',
                        'transit_stations',
                        'workplaces',
                        'residential',
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

    #Plot distribution
    fig,ax = plt.subplots(figsize=(9/2.54,9/2.54))
    plt.hist(np.log10(y_high+0.001),bins=20,alpha=0.5,label='high')
    plt.hist(np.log10(y_low+0.001),bins=20,alpha=0.5,label='low')
    plt.title('Target case disributions')
    plt.xlabel('log cases per 100000')
    plt.yticks([])
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir+'case_distr.png',format='png')
    plt.close()

    return X_high,y_high,X_low,y_low


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
def build_net(n1,n2,input_dim):
    '''Build the net using Keras
    '''
    z = L.Input((input_dim,), name="Patient")

    x1 = L.Dense(n1, activation="relu", name="d1")(z)
    x1 = L.BatchNormalization()(x1)
    x2 = L.Dense(n2, activation="relu", name="d2")(x1)
    x3 = L.Dense(n2, activation="relu", name="d3")(x2)

    p1 = L.Dense(3, activation="linear", name="p1")(x3)
    p2 = L.Dense(3, activation="relu", name="p2")(x3)
    preds = L.Lambda(lambda x: x[0] + tf.cumsum(x[1], axis=1),
                     name="preds")([p1, p2])

    model = M.Model(z, preds, name="Dense")
    model.compile(loss='mae', optimizer=tf.keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.01, amsgrad=False),metrics=['mae'])
    return model


#####MAIN#####
args = parser.parse_args()
#Seed
seed_everything(0) #The answer it is
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
threshold = args.threshold[0]
outdir = args.outdir[0]
#Use only data from start date
adjusted_data = adjusted_data[adjusted_data['Date']>=start_date]
#Get data
X_high,y_high,X_low,y_low =  get_features(adjusted_data,train_days,forecast_days,threshold,outdir)


#Convert to log for training
np.log(y_train+0.001)
#Get net parameters
BATCH_SIZE=64
EPOCHS=100
n1=X.shape[1] #Nodes layer 1
n2=X.shape[1 #Nodes layer 2

#Make net
net = build_net(n1,n2,X.shape[1]+1)
print(net.summary())


#KFOLD
NFOLD = 5
#kf = KFold(n_splits=NFOLD,shuffle=True, random_state=42)
train_split, val_split = kfold(len(X),NFOLD)
#Save errors
errors = []
corrs = []

for fold in range(NFOLD):
    tr_idx, val_idx = train_split[fold], val_split[fold]
    print('Number of valid points',len(val_idx))
    tensorboard = TensorBoard(log_dir=outdir+'fold'+str(fold))
    print("FOLD", fold)
    net = build_net(n1,n2,X.shape[1])
    #Data generation
    training_generator = DataGenerator(X[tr_idx], y[tr_idx], BATCH_SIZE)
    valid_generator = DataGenerator(X[val_idx], y[val_idx], BATCH_SIZE)

    net.fit(training_generator,
            validation_data=valid_generator,
            epochs=EPOCHS,
            callbacks = [tensorboard]
            )

    preds = net.predict(X[val_idx])
    preds[preds<0]=0
    errors.append(np.average(np.absolute(preds[:,1]-y[val_idx])))
    corrs.append(pearsonr(preds[:,1],y[val_idx])[0])
pdb.set_trace()
pdb.set_trace()
