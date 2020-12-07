#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--in_csv', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to input csv to be evaluated.')
parser.add_argument('--outdir', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to output directory. Include /in end')


def calc_error_pcc(pred,true):
    '''
    Evalute predictions
    '''

    #Calculate the cumulative error
    cumulative_error = np.sum(np.absolute(pred-true))

    #Evaluate PCC
    R,p = pearsonr(pred,true)

    return cumulative_error, R


#####MAIN#####
matplotlib.rcParams.update({'font.size': 7})
args = parser.parse_args()
in_csv = args.in_csv[0]
outdir = args.outdir[0]

#Define region
in_csv['GeoID'] = in_csv['CountryName'] + '__' + in_csv['RegionName'].astype(str)
for region in in_csv['GeoID'].unique():
    geo_data = in_csv[in_csv['GeoID']==region]
    pdb.set_trace()
