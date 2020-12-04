#! /usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Prints files with different combinations of parameters to be used to define different models.
'''


filters = [8,16,32,64]
dilation_rate = [3,6,12]
kernel_size = [5,10,15]
lr = [0.1,0.01,0.001]
num_convolutional_layers=[1,2,3]


for fil in filters:
	for dr in dilation_rate:
		for k in kernel_size:
			for l in lr:
				for nc in num_convolutional_layers:
					name = str(fil)+'_'+str(dr)+'_'+str(k)+'_'+str(l)+'_'+str(nc)+'.params'
					with open(name, "w") as file:
						file.write('filters='+str(fil)+'\n')
						file.write('dilation_rate='+str(dr)+'\n')
						file.write('kernel_size='+str(k)+'\n')
						file.write('lr='+str(l)+'\n')
						file.write('num_convolutional_layers='+str(nc)+'\n')
