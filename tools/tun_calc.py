#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 01:36:27 2020

@author: sdas
"""
import numpy as np

def calc_tun(data,activation_model,layer_names):
    tun_value = [[[] for j in range(len(layer_names))] for i in range(len(data))]
    for imgs,img_tensor in enumerate(data):
      img_tensor = img_tensor.reshape([1,224,224,3])
      intermediate_activations = activation_model.predict(img_tensor)
      for l in range(len(layer_names)):
        layer_activation = intermediate_activations[l]
        for k in range(layer_activation.shape[3]):
          tun_value[imgs][l].append(np.mean(layer_activation[0,:,:,k]))
    return tun_value


def calc_avg(tun_value):
    avg_tun_value = [[0 for item in subl] for subl in tun_value[0]]
    for img in range(len(tun_value)):
      tensor_tun = tun_value[img]
      for layer in range(len(tensor_tun)):
        for map in range(len(tensor_tun[layer])):
          avg_tun_value[layer][map] += tensor_tun[layer][map]
    
    avg_tun_value = [[item / len(tun_value) for item in subl] for subl in avg_tun_value]
    
    return avg_tun_value

def calc_std(tun_value,avg_tun_value):
    std_tun_value = [[0.0 for item in subl] for subl in tun_value[0]]
    for img in range(len(tun_value)):
      tensor_tun = tun_value[img]
      for layer in range(len(tensor_tun)):
        for mapa in range(len(tensor_tun[layer])):
          std_tun_value[layer][mapa] += (tensor_tun[layer][mapa] - avg_tun_value[layer][mapa])**2
          
    std_tun_value = [[(item / len(tun_value))**(1/2) for item in subl] for subl in avg_tun_value]