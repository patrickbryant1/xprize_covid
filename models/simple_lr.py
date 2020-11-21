#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import sys
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


import pdb
#Arguments for argparse module:
parser = argparse.ArgumentParser(description = '''Simple linear regression model.''')

parser.add_argument('--processed_data', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to processed data file.')

parser.add_argument('--outdir', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to output directory. Include /in end')


def get_features():
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
                        'rescaled_cases',
                        'cumulative_rescaled_cases',
                        'death_to_case_scale',
                        'case_death_delay']
                        
    sel = processed_data[selected_features]

    pdb.set_trace()

def train_test_split()
