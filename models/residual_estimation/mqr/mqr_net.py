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
parser = argparse.ArgumentParser(description = '''A multiple Quantile regression model for residual estimation.''')
parser.add_argument('--days_ahead', nargs=1, type= int,
                  default=sys.stdin, help = 'Number of days ahead to fit')
parser.add_argument('--X_train', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to predictions of train data.')
parser.add_argument('--train_preds', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to predictions of train data.')
parser.add_argument('--residuals', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to true values of train data.')
#parser.add_argument('--param_combo', nargs=1, type= int, default=sys.stdin, help = 'Parameter combo.')
parser.add_argument('--outdir', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to output directory. Include /in end')

#######FUNCTIONS#######
def seed_everything(seed=2020):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

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
def build_net(n1,n2,input_dim):
    '''Build the net using Keras
    '''
    z = L.Input((input_dim,), name="Patient")

    x1 = L.Dense(n1, activation="relu", name="d1")(z)
    x1 = L.BatchNormalization()(x1)
    x2 = L.Dense(n2, activation="relu", name="d2")(x1)
    x2 = L.BatchNormalization()(x2)

    p1 = L.Dense(3, activation="linear", name="p1")(x2)
    p2 = L.Dense(3, activation="relu", name="p2")(x2)
    preds = L.Lambda(lambda x: x[0] + tf.cumsum(x[1], axis=1),
                     name="preds")([p1, p2])

    model = M.Model(z, preds, name="Dense")
    model.compile(loss=qloss, optimizer=tf.keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.01, amsgrad=False),metrics=['mae'])
    return model


#####MAIN#####
#Set font size
matplotlib.rcParams.update({'font.size': 7})
args = parser.parse_args()
days_ahead = args.days_ahead[0]
X_train = np.load(args.X_train[0],allow_pickle=True)
train_preds = np.load(args.train_preds[0],allow_pickle=True)
residuals = np.load(args.residuals[0],allow_pickle=True)
outdir = args.outdir[0]

#Seed
seed_everything(42) #The answer it is

#Get net parameters
BATCH_SIZE=256
EPOCHS=200
n1=16 #Nodes layer 1
n2=16 #Nodes layer 2

#Make net
net = build_net(n1,n2,X_train.shape[1]+1)
print(net.summary())
#KFOLD
NFOLD = 5
kf = KFold(n_splits=NFOLD)
fold=0
pdb.set_trace()
for tr_idx, val_idx in kf.split(X_train):
    fold+=1
    tensorboard = TensorBoard(log_dir=outdir+'fold'+str(fold))
    print("FOLD", fold)
    net = build_net(n1,n2,X_train.shape[1],bins)
    #Data generation
    training_generator = DataGenerator(X_train[tr_idx], y_train[tr_idx], BATCH_SIZE)
    valid_generator = DataGenerator(X_train[val_idx], y_train[val_idx], BATCH_SIZE)

    net.fit(training_generator,
            validation_data=valid_generator,
            epochs=EPOCHS,
            callbacks = [tensorboard]
            )

    #Test the net
    test(net, X_test,y_test,populations,regions)
    pdb.set_trace()
pdb.set_trace()
