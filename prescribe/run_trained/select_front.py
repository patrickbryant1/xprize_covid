#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--population', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to input np array with entire population.')
parser.add_argument('--total_front', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to input np array with entire pareto front.')
parser.add_argument('--selected_front', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to input str with selected front.')
parser.add_argument('--outdir', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to output directory. Include /in end')


def select_top10(total_front,selected_front,population,outdir):
    '''Select the pareto front
    '''


    for i in range(len(total_front)):
        plt.scatter(total_front[i,0], total_front[i,1], c="b")
        plt.text(total_front[i,0], total_front[i,1],str(i), fontsize=12)
        if i in selected_front:
            plt.scatter(total_front[i,0], total_front[i,1], c="r")

    plt.xlabel('Cases')
    plt.ylabel('Stringency')
    plt.title('Pareto front')
    plt.savefig(outdir+'sel_front.png',dpi=300)
    plt.close()
    pdb.set_trace()

    sel_pop = population[selected_front]
    np.save(outdir+'selected_population.npy',sel_pop)

#####MAIN#####
matplotlib.rcParams.update({'font.size': 7})
args = parser.parse_args()
population = np.load(args.population[0],allow_pickle=True)
total_front = np.load(args.total_front[0],allow_pickle=True)
selected_front = np.array(args.selected_front[0].split(','),dtype='int32')
outdir = args.outdir[0]
select_top10(total_front,selected_front,population,outdir)
pdb.set_trace()
