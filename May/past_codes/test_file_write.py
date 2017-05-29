#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 22:57:29 2017

@author: hsadeghi
"""
#
results={};
results['index'            ] = 1
results['training_error'    ]= 2
results['test_error'        ]= 3


results['input_dim']         = 4
results['learning_rate' ]    = 5
                               
save_name = "/vol/grid-solar/sgeusers/hsadeghi/config_{}_{}.txt".format(2, 3) 
file_1 = open(save_name, "w") 

for value,key in enumerate(results):
    
    file_1.write( key + '   '+ str(value)+ '\n')

file_1.close()
