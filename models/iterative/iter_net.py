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
from math import e

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import tensorflow.keras.layers as L
import tensorflow.keras.models as M
from tensorflow.keras.callbacks import TensorBoard
from scipy.stats import pearsonr


import pdb
#Arguments for argparse module:
parser = argparse.ArgumentParser(description = '''A dense iterative regression model.''')

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
parser.add_argument('--num_pred_periods', nargs=1, type= int,
                  default=sys.stdin, help = 'Number of prediction periods.')
parser.add_argument('--threshold', nargs=1, type= float,
                  default=sys.stdin, help = 'Threshold.')
parser.add_argument('--outdir', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to output directory. Include /in end')

#######FUNCTIONS#######
def get_features(adjusted_data,train_days,forecast_days,num_pred_periods,t,outdir):
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
                        'monthly_temperature',
                        'smoothed_cases',
                        'cumulative_smoothed_cases',
                        #'rescaled_cases',
                        #'cumulative_rescaled_cases',
                        'death_to_case_scale',
                        'case_death_delay',
                        'gross_net_income',
                        'population_density',
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
        X1 = np.load(outdir+'X1.npy', allow_pickle=True)
        X2 = np.load(outdir+'X3.npy', allow_pickle=True)
        X3 = np.load(outdir+'X3.npy', allow_pickle=True)
        y = np.load(outdir+'y.npy', allow_pickle=True)



    except:
        sel = adjusted_data[selected_features]
        X1,X2,X3,y = split_for_training(sel,train_days,forecast_days,num_pred_periods)

        #Save
        np.save(outdir+'X1.npy',X1)
        np.save(outdir+'X2.npy',X2)
        np.save(outdir+'X3.npy',X3)
        np.save(outdir+'y.npy',y)

    #Split into high and low

    return X1,X2,X3,y


def split_for_training(sel,train_days,forecast_days,num_pred_periods):
    '''Split the data for training and testing
    '''
    X1 = [] #Input periods
    X2 = []
    X3 = []
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
             'Share of Deaths from Air Pollution (%)','CO2 emissions (metric tons per capita)', 'Air transport (# carrier departures worldwide)','population',
             'cumulative_smoothed_cases'})

            #Normalize the cases by 100'000 population
            #country_region_data['rescaled_cases']=country_region_data['rescaled_cases']/(population/_high000)
            #country_region_data['cumulative_rescaled_cases']=country_region_data['cumulative_rescaled_cases']/(population/_high000)
            country_region_data['smoothed_cases']=country_region_data['smoothed_cases']/(population/100000)
            #country_region_data['cumulative_smoothed_cases']=country_region_data['cumulative_smoothed_cases']/(population/100000)
            #Loop through and get the data

            for di in range(len(country_region_data)-(train_days*num_pred_periods+forecast_days-1)):

                #Get all features
                all_xi = []
                all_yi = []
                for pi in range(num_pred_periods):
                    xi = np.array(country_region_data.loc[di+train_days*pi:di+train_days*(pi+1)-1])
                    case_medians = np.median(xi[:,13:],axis=0)
                    xi = np.average(xi,axis=0)
                    xi[13:]=case_medians
                    all_xi.append(xi)

                    #Get median
                    yi = np.array(country_region_data.loc[di+train_days*(pi+1):di+train_days*(pi+2)+forecast_days-1]['smoothed_cases'])
                    all_yi.append(np.median(yi))





                #Add
                X1.append(np.append([death_to_case_scale,case_death_delay,gross_net_income,population_density,
                                    pdi, idv, mas, uai, ltowvs, ivr,upop, pop65, gdp, obesity,
                                    cancer, smoking_deaths, pneumonia_dr, air_pollution_deaths, co2_emission,
                                    air_transport, population],all_xi[0].flatten()))

                X2.append(np.append([death_to_case_scale,case_death_delay,gross_net_income,population_density,
                                    pdi, idv, mas, uai, ltowvs, ivr,upop, pop65, gdp, obesity,
                                    cancer, smoking_deaths, pneumonia_dr, air_pollution_deaths, co2_emission,
                                    air_transport, population],all_xi[1].flatten()[:-1]))
                X3.append(np.append([death_to_case_scale,case_death_delay,gross_net_income,population_density,
                                    pdi, idv, mas, uai, ltowvs, ivr,upop, pop65, gdp, obesity,
                                    cancer, smoking_deaths, pneumonia_dr, air_pollution_deaths, co2_emission,
                                    air_transport, population],all_xi[2].flatten()[:-1]))


                y.append(np.array(all_yi))


    return np.array(X1),np.array(X2),np.array(X3), np.array(y)


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
    def __init__(self, X1_train_fold,X2_train_fold,X3_train_fold, y_train_fold, batch_size=1, shuffle=True):
        'Initialization'
        self.X1_train_fold = X1_train_fold
        self.X2_train_fold = X2_train_fold
        self.X3_train_fold = X3_train_fold
        self.y_train_fold = y_train_fold
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.X1_train_fold) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        batch_indices = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        #domain_index = np.take(range((len(self.X_train_fold))),indexes)

        # Generate data
        X1_batch, X2_batch, X3_batch, y_batch = self.__data_generation(batch_indices)

        return [X1_batch, X2_batch, X3_batch], y_batch

    def on_epoch_end(self): #Will be done at epoch 0 also
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.X1_train_fold))
        np.random.shuffle(self.indexes)


    def __data_generation(self, batch_indices):
        'Generates data containing batch_size samples'

        return self.X1_train_fold[batch_indices],self.X2_train_fold[batch_indices],self.X3_train_fold[batch_indices],self.y_train_fold[batch_indices]

#####LOSSES AND SCORES#####

#Custom loss
#============================#

def bin_loss(y_true, y_pred):
    tf.dtypes.cast(y_true, tf.float32)
    tf.dtypes.cast(y_pred, tf.float32)
    #Shold make this a log loss
    g_loss = tf.keras.losses.MeanAbsoluteError()(y_true, y_pred) #general, compare difference
    #log_g_loss = keras.backend.log(g_loss/100)
    #Gauss for loss
    #gauss = keras.backend.random_normal_variable(shape=(batch_size, 1), mean=0.7, scale=0.3) # Gaussian distribution, scale: Float, standard deviation of the normal distribution.
    kl_loss = keras.losses.kullback_leibler_divergence(y_true, y_pred) #better than comparing to gaussian


    # #Normalize due to proportion
    # kl_p = sum_kl_loss/(sum_g_loss+sum_kl_loss)
    # g_p = sum_g_loss/(sum_g_loss+sum_kl_loss)

    # sum_kl_loss = sum_kl_loss/kl_p
    # sum_g_loss = sum_g_loss/g_p
    loss = K.abs(g_loss) #+ K.abs(kl_loss)
    #Scale with R? loss = loss/R - on_batch_end
    #Normalize loss by percentage contributions: divide by contribution
    #Write batch generator to avoid incompatibility in shapes
    #problem at batch end due to kongruens
    return K.mean(loss)

#####BUILD NET#####
def build_net(n1,n2,dim1,dim2,dim3):
    '''Build the net using Keras
    '''
    shared_dense1 = L.Dense(n1, activation="relu", name="d1")
    shared_dense2 = L.Dense(n2, activation="relu", name="d2")
    shared_dense3 = L.Dense(1, activation="linear", name="d3")

    inp1 = L.Input((dim1,), name="inp1") #Inputs with median cases
    inp2 = L.Input((dim2,), name="inp2") #Inputs without median cases
    inp3 = L.Input((dim3,), name="inp3") #Inputs without median cases

    pred_step1 = L.BatchNormalization()(shared_dense3(shared_dense2(shared_dense1(inp1))))

    #Concat preds 1 with inp 2
    pred_in2 = L.Concatenate()([inp2,pred_step1])
    pred_step2 = L.BatchNormalization()(shared_dense3(shared_dense2(shared_dense1(pred_in2))))
    #Concat preds 2 with inp 3
    pred_in3 = L.Concatenate()([inp3,pred_step2])
    pred_step3 = L.BatchNormalization()(shared_dense3(shared_dense2(shared_dense1(pred_in3))))

    #Attentions
    attention1 = L.Attention(name="a1")([shared_dense1(inp1),shared_dense1(inp1)])
    attention2 = L.Attention(name="a2")([shared_dense1(pred_in2),shared_dense1(pred_in2)])
    attention3 = L.Attention(name="a3")([shared_dense1(pred_in3),shared_dense1(pred_in3)])
    preds = L.Concatenate()([pred_step1,pred_step2,pred_step3])

    model = M.Model([inp1,inp2,inp3], preds, name="Dense")
    model.compile(loss=bin_loss, optimizer=tf.keras.optimizers.Adam(lr=0.01, decay=0.001,),metrics=['mae'])
    return model

def fit_data(X1,X2,X3,y,mode,outdir):

    #Get inpu2 - no cases

    #KFOLD
    NFOLD = 5
    #kf = KFold(n_splits=NFOLD,shuffle=True, random_state=42)
    train_split, val_split = kfold(len(X1),NFOLD)
    #Save errors
    errors = []
    corrs = []

    #Get net parameters
    BATCH_SIZE=256
    EPOCHS=1000
    n1=X1.shape[1] #Nodes layer 1
    n2=X1.shape[1] #Nodes layer 2

    #Make net
    net = build_net(n1,n2,X1.shape[1],X2.shape[1],X3.shape[1])
    #serialize model to JSON
    model_json = net.to_json()
    with open(outdir+"model.json", "w") as json_file:
    	json_file.write(model_json)
    print('Saved model...')
    print(net.summary())


    for fold in range(NFOLD):
        tr_idx, val_idx = train_split[fold], val_split[fold]
        print('Number of valid points',len(val_idx))
        tensorboard = TensorBoard(log_dir=outdir+'fold'+str(fold+1))
        #Checkpoint
        filepath=outdir+"fold_"+str(fold+1)+"_weights-{epoch:02d}-.hdf5"
        checkpoint = keras.callbacks.ModelCheckpoint(filepath, verbose=1, save_best_only=True, mode='min')

        #Build net
        net =build_net(n1,n2,X1.shape[1],X2.shape[1],X3.shape[1])
        #Data generation
        training_generator = DataGenerator(X1[tr_idx], X2[tr_idx], X3[tr_idx], y[tr_idx], BATCH_SIZE)
        valid_generator = DataGenerator(X1[val_idx], X2[val_idx], X3[val_idx], y[val_idx], BATCH_SIZE)
        #Fit net
        print("Fitting fold", fold+1)
        net.fit(training_generator,
                validation_data=valid_generator,
                epochs=EPOCHS,
                callbacks = [tensorboard, checkpoint]
                )

        #Look at activations
        #get_activations(net, [X1[val_idx],X2[val_idx], X3[val_idx]])

        preds = net.predict([X1[val_idx],X2[val_idx], X3[val_idx]])
        plt.hist(preds,color=['b','b','b'],alpha=0.5)
        plt.hist(y[val_idx],color=['r','r','r'],alpha=0.5)
        plt.savefig(outdir+'hist_fold'+str(fold+1)+'.png',format='png')
        plt.close()
        #preds = np.power(e,preds)
        #true = np.power(e,y[val_idx])

        true = y[val_idx]

        for i in range(preds.shape[1]):
            errors.append(np.average(np.absolute(preds[:,i]-true[:,i])))
            corrs.append(pearsonr(preds[:,i],true[:,i])[0])
            plt.scatter(preds[:,i],true[:,i],label='Median '+str(i+1),s=1,alpha=0.5)
        plt.legend()
        plt.savefig(outdir+'median_'+str(fold+1)+'.png',format='png')
        plt.close()



def get_activations(net, data):

    #Get layer output
    #layers 1-3 are the dense ones
    get_1st_layer_output = K.function([net.layers[0].input],[net.layers[1].output])
    get_2nd_layer_output = K.function([net.layers[0].input],[net.layers[2].output])
    get_3d_layer_output = K.function([net.layers[0].input],[net.layers[3].output])
    layer_output1 = get_1st_layer_output(data)[0]
    layer_output2 = get_2nd_layer_output(data)[0]
    layer_output3 = get_3d_layer_output(data)[0]

    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0) # only difference

    def attention(act):
        #Dot
        dot = np.dot(act.T,act)
        dot = dot/np.max(dot)
        soft = softmax(dot)
        att = np.dot(soft,act.T)
        plt.bar(np.arange(att.shape[0]),np.average(att,axis=1))
        plt.show()
    attention(layer_output1)
    attention(layer_output2)
    attention(layer_output3)



#####MAIN#####
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
num_pred_periods = args.num_pred_periods[0]
threshold = args.threshold[0]
outdir = args.outdir[0]
#Use only data from start date
adjusted_data = adjusted_data[adjusted_data['Date']>=start_date]
#Exclude the regional data from Brazil
exclude_index = adjusted_data[(adjusted_data['CountryCode']=='BRA')&(adjusted_data['RegionCode']!='0')].index
adjusted_data = adjusted_data.drop(exclude_index)
#Get data
X1,X2,X3,y = get_features(adjusted_data,train_days,forecast_days,num_pred_periods,threshold,outdir)

#Fit high
#Convert to log for training
fit_data(X1,X2,X3,np.log(y+0.001),'high',outdir)

pdb.set_trace()
pdb.set_trace()
