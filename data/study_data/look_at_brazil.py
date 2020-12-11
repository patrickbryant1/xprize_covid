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

#Use only data from start date
start_date = '2020-11-01'
adjusted_data = adjusted_data[adjusted_data['Date']>=start_date]
pdb.set_trace()
