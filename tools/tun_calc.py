#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 01:36:27 2020

@author: sdas
"""
import numpy as np


def calc_avgtun(train_it,activation_model,layer_names,):
    tun_value = np.zeros((len(layer_names),600))
    for img_tensor in train_it:
      img_tensor = img_tensor.reshape([1,224,224,3])
      intermediate_activations = activation_model.predict(img_tensor)
      for l in range(len(layer_names)):
        layer_activation = intermediate_activations[l]
        for k in range(layer_activation.shape[3]):
          tun_value[l,k] += np.mean(layer_activation[0,:,:,k])
          
    return tun_value
