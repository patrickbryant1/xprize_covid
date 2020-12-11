#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import sys
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.metrics import mutual_info_score
from scipy.stats import pearsonr
from scipy import stats
import seaborn as sns
import numpy as np



import pdb
#Arguments for argparse module:
parser = argparse.ArgumentParser(description = '''Stude data relationships.''')

parser.add_argument('--oxford_file', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to processed data file.')
parser.add_argument('--regional_populations', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to other regional populations (UK).')
parser.add_argument('--country_populations', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to country populations.')
parser.add_argument('--outdir', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to output directory. Include /in end')



#####MAIN#####
#Set font size
matplotlib.rcParams.update({'font.size': 7})
args = parser.parse_args()
oxford_data = pd.read_csv(args.oxford_file[0],
                 parse_dates=['Date'],
                 encoding="ISO-8859-1",
                 dtype={"RegionName": str,
                        "RegionCode": str},
                 error_bad_lines=False)
regional_populations = pd.read_csv(args.regional_populations[0])
country_populations = pd.read_csv(args.country_populations[0])
outdir = args.outdir[0]
oxford_data = oxford_data.fillna(0) #Replace NaN
#Use only data from start date
start_date = '2020-11-01'
oxford_data  = oxford_data [oxford_data ['Date']>=start_date]
brazil_data = oxford_data[oxford_data['CountryCode']=='BRA']

fig,ax = plt.subplots(figsize=(10,7.5))
for region in brazil_data.RegionName.unique():
    region_data = brazil_data[brazil_data['RegionName']==region]
    if region==0:
        region = 'Whole Brazil'
    plt.scatter(region_data['Date'],region_data['ConfirmedCases'],label=region)
plt.legend()
plt.yscale('log')
plt.ylabel('cases')
plt.title("Brazil's reported cases - all regions")
plt.tight_layout()
plt.savefig(outdir+'brazil_cases.png', format='png')
plt.close()


#Look at population normalized
fig,ax = plt.subplots(figsize=(10,7.5))
for region in brazil_data.RegionCode.unique():
    region_data = brazil_data[brazil_data['RegionName']==region]
    if region in regional_populations['Region Code'].values:
        population=regional_populations[regional_populations['Region Code']==region]['2019 population'].values[0]
    else:
        population=country_populations[country_populations['Country Code']==region_data['CountryCode'].values[0]]['2018'].values[0]
    if region==0:
        region = 'Whole Brazil'
    plt.scatter(region_data['Date'],region_data['ConfirmedCases']/(population/100000),label=region)

plt.legend()
plt.ylabel('Cases per 100000 population')
plt.title("Brazil's reported cases - all regions")
plt.tight_layout()
plt.savefig(outdir+'brazil_norm_cases.png', format='png')
plt.close()
