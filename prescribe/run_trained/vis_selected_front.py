#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--selected_front', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to input str with selected front.')
parser.add_argument('--outdir', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to output directory. Include /in end')


#####MAIN#####
matplotlib.rcParams.update({'font.size': 7})
args = parser.parse_args()
selected_front = np.load(args.selected_front[0],allow_pickle=True)
outdir = args.outdir[0]

#Plot the selected front
#The selected front contains 10 models, each with (12,2,2) = 48 weights
#The prescription works like this:
##Multiply prev ip with the 2 prescr weight layers of the individual
#prescr = prev_ip*ip_weights*individual[:,0,0]*individual[:,0,1]
#Add the case focus
#prescr += np.array([prev_cases]).T*individual[:,1,0]*individual[:,1,1]
#This means that the first
prescr_weights1 = []
prescr_weights2 = []
case_weights1 = []
case_weights2 = []
for i in range(len(selected_front)):
    individual = np.reshape(selected_front[i],(12,2,2))
    prescr_weights1.append(individual[:,0,0])
    prescr_weights2.append(individual[:,0,1])
    case_weights1.append(individual[:,1,0])
    case_weights2.append(individual[:,1,1])
#Plot
def plot_weights(weights,xtick_labels,title,outdir):
    fig,ax = plt.subplots(figsize=(9/2.54,9/2.54))
    plt.imshow(np.array(weights))
    plt.yticks(np.arange(10),np.arange(1,11))
    plt.xticks(np.arange(0,12),xtick_labels)
    plt.ylabel('PrescriptionIndex')
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(outdir+title+'.png',format='png',dpi=300)
    plt.close()

NPI_order = ['C1','C2','C3','C4','C5','C6','C7','C8','H1','H2','H3','H6']
plot_weights(prescr_weights1,NPI_order,'NPI weights 1',outdir)
plot_weights(prescr_weights2,NPI_order,'NPI weights 2',outdir)
plot_weights(case_weights1,NPI_order,'Case weights 1',outdir)
plot_weights(case_weights2,NPI_order,'Case weights 2',outdir)
