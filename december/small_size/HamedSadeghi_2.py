#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 11:09:10 2017

@author: hsadeghi
"""

import numpy as np

#%%
class image_database:
    
    
    def __init__(self, D, K):
        #"Assume D is dictionary of size num_imgs and each key (0,..,num_imgs-1) has a matrix (N x M) as its value"
        # here we build a hash map
        
        self.database = D # we assume it is not empty
        self.K = K
        self.num_imgs = len(D)
        self.M = len(D[0])
        self.N = len(D[0][0])
        self.hash_map = {}
        
        self.height = self.N//self.K
        self.width = self.M//self.K
        
        
        for i in range(self.num_imgs):
            I =  self.database[i]
            self.hash_map[i] = image_database.calculate_hash_key(I)

#%%
    def calculate_hash_key(self, I):

            I = np.array(I)  # to use splitting abilities of np            
            hash_key = '' # of teh entire image
            
            for k in range(self.K - 1):
                for kk in range(self.K - 1):
                    sub_I = I[k*self.height : (k + 1) *self.height,
                              kk * self.width : (kk + 1) *self.width]
                    
                    sub_I = np.reshape(sub_I, (1, -1)) #vectorizing
                    sub_I = str(sub_I) 
                    
                    hash_key += sub_I
                    hash_key = int(hash_key, 2) #converts the string as binary (base 2) rep to decimal,
                    # returns an integer
                    
            return hash_key
        
#%%

    def restore(self, I):
        
        hash_key =  image_database.calculate_hash_key(I)
        self.hash_map[hash_key] = True
       
        
    #%%
    def retrive(self, I): # return True if exists or -1 otherwise 
           
        hash_key  =  image_database.calculate_hash_key(I)
        
        if hash_key in self.hash_map:
            return True
        else:
            return -1
    
    

###################################
    def forms_complete_image(self, seq):
        "incomplete"
        
                
        
    