#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import pdb
#Arguments for argparse module:
parser = argparse.ArgumentParser(description = '''Preprocess data for training.''')

parser.add_argument('--oxford_file', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to oxford data file.')

parser.add_argument('--outdir', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to output directory. Include /in end')

def parse_regions(oxford_data):
    '''Parse and encode all regions
    '''

    #Define country and regional indices
    oxford_data['Country_index']=0
    oxford_data['Region_index']=0
    country_codes = oxford_data['CountryCode'].unique()
    ci = 0 #Country index
    for cc in country_codes:
        #Create fig for vis
        
        ri = 0 #Region index
        country_data = oxford_data[oxford_data['CountryCode']==cc]
        #Set index
        oxford_data.at[country_data.index,'Country_index']=ci
        #Get regions
        regions = country_data['RegionCode'].dropna().unique()
        #Check if regions
        if regions.shape[0]>0:
            for region in regions:
                country_region_data = country_data[country_data['RegionCode']==region]
                oxford_data.at[country_region_data.index,'Country_index']=ri
                #Icrease ri
                ri+=1
        #Increase ci
        ci+=1

    pdb.set_trace()

#####MAIN#####
args = parser.parse_args()
oxford_data = pd.read_csv(args.oxford_file[0],
                 parse_dates=['Date'],
                 encoding="ISO-8859-1",
                 dtype={"RegionName": str,
                        "RegionCode": str},
                 error_bad_lines=False)
outdir = args.outdir[0]


parse_regions(oxford_data)
pdb.set_trace()
