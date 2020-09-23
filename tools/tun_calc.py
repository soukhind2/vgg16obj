#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 01:36:27 2020

@author: sdas
"""
import numpy as np

def calc_tun(data,activation_model,layer_names):
    tun_value = np.zeros((len(data),len(layer_names),600))
    for imgs,img_tensor in enumerate(data):
      img_tensor = img_tensor.reshape([1,224,224,3])
      intermediate_activations = activation_model.predict(img_tensor)
      for l in range(len(layer_names)):
        layer_activation = intermediate_activations[l]
        for k in range(layer_activation.shape[3]):
          tun_value[imgs,l,k] += np.mean(layer_activation[0,:,:,k])
          
    return tun_value
