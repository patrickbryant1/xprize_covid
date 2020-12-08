#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import sys
import os
import numpy as np
import random
import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns

import pdb
#Arguments for argparse module:
parser = argparse.ArgumentParser(description = '''A CNN regression model.''')

parser.add_argument('--params_file', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to file with params.')
parser.add_argument('--params_order', nargs=1, type= str,
                  default=sys.stdin, help = 'Order of params.')
parser.add_argument('--results_dir', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to results.')
parser.add_argument('--outdir', nargs=1, type= str,
                  default=sys.stdin, help = 'Outdir.')

def parse_params(params_file, params_order):
    '''Parse the parameter combinations
    '''
    param_combos = {}
    for key in params_order:
        param_combos[key]=[]
    with open(params_file, 'r') as file:
        for line in file:
            line = line.rstrip()
            line = line[:-7].split('_')
            for i in range(len(line)):
                param_combos[params_order[i]].append(float(line[i]))

    param_df = pd.DataFrame.from_dict(param_combos)
    return param_df


def get_results(param_df, results_dir):
    '''Get results
    '''
    combos = glob.glob(results_dir+'COMBO*')
    train_loss = []
    train_std = []
    val_loss = []
    val_std = []
    for i in range(len(combos)):
        tr = np.load(results_dir+'COMBO'+str(i+1)+'/train_errors.npy',allow_pickle=True)
        val = np.load(results_dir+'COMBO'+str(i+1)+'/train_errors.npy',allow_pickle=True)

        #Take min from each fold
        train_loss.append(np.average(np.min(tr,axis=1)))
        train_std.append(np.std(np.min(tr,axis=1)))
        val_loss.append(np.average(np.min(val,axis=1)))
        val_std.append(np.std(np.min(val,axis=1)))

    param_df['train_mae'] = train_loss
    param_df['train_std'] = train_std
    param_df['val_mae'] = val_loss
    param_df['val_std'] = val_std
    pdb.set_trace()


#####MAIN#####
args = parser.parse_args()
params_file = args.params_file[0]
params_order = args.params_order[0].split(',')
results_dir = args.results_dir[0]
outdir = args.outdir[0]

#Parse params
param_df = parse_params(params_file, params_order)
#Get results
get_results(param_df, results_dir)
