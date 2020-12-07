#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import sys
import os
import numpy as np
import random
import pandas as pd

import pdb
#Arguments for argparse module:
parser = argparse.ArgumentParser(description = '''A CNN regression model.''')

parser.add_argument('--params_file', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to file with params.')
parser.add_argument('--params_order', nargs=1, type= str,
                  default=sys.stdin, help = 'Order of params.')
parser.add_argument('--outdir', nargs=1, type= str,
                  default=sys.stdin, help = 'Outdir.')





#####MAIN#####
args = parser.parse_args()
params_file = args.params_file[0]
params_order = args.params_order[0].split('')
