#! /usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Prints files with different combinations of parameters to be used to define different models.
'''


filters = [8,16,32,64]
dilation_rate = [3,6,12]
lr = [0.1,0.01,0.001]


for fil in filters:
	for dr in dilation_rate:
		for l in lr:
			name = str(fil)+'_'+str(dr)+'_'+str(l)+'.params'
			with open(name, "w") as file:
				file.write('filters='+str(fil)+'\n')
				file.write('dilation_rate='+str(dr)+'\n')
				file.write('lr='+str(l)+'\n')
