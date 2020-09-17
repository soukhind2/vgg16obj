#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 23:40:05 2020

@author: sdas


This script is used to initialize and build the model for the stem and the two
branches of the top model. All dense layers are ReLu activated.

input_shape: Shape of the input tensor
b1_len: The size of each of the dense layers for branch 1. Input the subsequent
sizes as an array.
b2_len: The sizes of each of the dense layers for branch 2.Input the subsequent
sizes as an array.


"""


from keras.models import Model
from keras.layers import Flatten, Dense, Dropout
from keras.layers import Input

class modelarch():
    
    def build_branch1(self, inputs , w , num_c = 2):
    
        
        for i,j in enumerate(w):
            if i == 0:
                b1 = Dense(j, activation="relu")(inputs)
                
            b1 = Dense(j, activation="relu")(b1)
                
        o1 = Dense(num_c,activation="softmax",name="branch1")(b1)
        
        return o1
    
    def build_branch2(self, inputs , w , num_c = 3):
        
        for i,j in enumerate(w):
            if i == 0:
                b2 = Dense(j, activation="relu")(inputs)
                
            b2 = Dense(j, activation="relu")(b2)
                
        o2 = Dense(num_c,activation="softmax",name="branch2")(b2)
        
        return o2
    
    def build_stem(self,inputs):
        
        x = Flatten()(inputs)
        x = Dense(4096, activation="relu")(x)
        x = Dense(4096, activation="relu")(x)
        
        return x
    
    def build_full_model(self,input_shape,b1_len,b2_len):
        
        inputs = Input(shape = input_shape)
        stem = self.build_stem(inputs)
        b1_out = self.build_branch1(stem,b1_len)
        b2_out = self.build_branch2(stem,b2_len)
        
        model = Model(inputs=inputs,
                      outputs = [b1_out, b2_out],
                      name="objnet")
        
        return model
    
'''
m = modelarch().build_full_model([7,7,512] ,
                                 stem_len = [4096,4096],
                                 b1_len = [4096,4096],
                                 b2_len = [4096,4096]
                                 )
'''




