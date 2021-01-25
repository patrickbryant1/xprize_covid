#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
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
in_csv = pd.read_csv(args.in_csv[0],
                          parse_dates=['Date'],
                          encoding="ISO-8859-1",
                          dtype={"RegionName": str},
                          error_bad_lines=True)
outdir = args.outdir[0]

NPI_COLS=['C1_School closing', 'C2_Workplace closing', 'C3_Cancel public events',
       'C4_Restrictions on gatherings', 'C5_Close public transport',
       'C6_Stay at home requirements', 'C7_Restrictions on internal movement',
       'C8_International travel controls', 'H1_Public information campaigns',
       'H2_Testing policy', 'H3_Contact tracing', 'H6_Facial Coverings']
#Define region
in_csv['GeoID'] = np.where(in_csv["RegionName"].isnull(),
                              in_csv["CountryName"],
                             in_csv["CountryName"] + ' / ' + in_csv["RegionName"])
#Evaluate all regions and save
stringency = np.zeros(len(in_csv.PrescriptionIndex.unique()))
for region in in_csv['GeoID'].unique():
    geo_data = in_csv[in_csv['GeoID']==region]
    for pi in geo_data.PrescriptionIndex.unique():
        geo_prescr = geo_data[geo_data.PrescriptionIndex==pi][NPI_COLS].values
        stringency[pi-1]+=np.sum(geo_prescr)


plt.bar(np.arange(len(stringency)),stringency)
plt.ylabel('Cumulative stringency')
plt.show()
pdb.set_trace()
