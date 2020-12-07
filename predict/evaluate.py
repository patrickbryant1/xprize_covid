#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

import pdb

def calc_error_pcc(pred,true):
    '''
    Evalute predictions
    '''

    #Calculate the cumulative error
    cumulative_error = np.sum(np.absolute(pred-true))

    #Evaluate PCC
    R,p = pearsonr(pred,true)

    return cumulative_error, R
