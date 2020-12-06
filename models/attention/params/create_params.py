#! /usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Prints files with different combinations of parameters to be used to define different models.
'''


num_nodes = [8,16,32,64]
lr = [0.1,0.01,0.001]
#num_layers=[1,2,3]

for n in num_nodes:
	for l in lr:
		name = str(n)+'_'+str(l)+'.params'
		with open(name, "w") as file:
			file.write('num_nodes='+str(n)+'\n')
			file.write('lr='+str(l)+'\n')
