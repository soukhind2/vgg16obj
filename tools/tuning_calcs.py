#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 01:36:27 2020

@author: sdas
"""
import numpy as np
import matplotlib.pyplot as plt

def calc_tun(data,activation_model,layer_names):
    tun_activ = [[[] for j in range(len(layer_names))] for i in range(len(data))]
    for imgs,img_tensor in enumerate(data):
      img_tensor = img_tensor.reshape([1,224,224,3])
      intermediate_activations = activation_model.predict(img_tensor)
      for l in range(len(layer_names)):
        layer_activation = intermediate_activations[l]
        for k in range(layer_activation.shape[3]):
          tun_activ[imgs][l].append(np.mean(layer_activation[0,:,:,k]))
    return tun_activ


def calc_avg(tun_activ):
    avg_tun_activ = [[0 for item in subl] for subl in tun_activ[0]]
    for img in range(len(tun_activ)):
      tensor_tun = tun_activ[img]
      for layer in range(len(tensor_tun)):
        for map in range(len(tensor_tun[layer])):
          avg_tun_activ[layer][map] += tensor_tun[layer][map]
    
    avg_tun_activ = [[item / len(tun_activ) for item in subl] for subl in avg_tun_activ]
    
    return avg_tun_activ

def calc_std(tun_activ,avg_tun_activ):
    std_tun_activ = [[0.0 for item in subl] for subl in tun_activ[0]]
    for img in range(len(tun_activ)):
      tensor_tun = tun_activ[img]
      for layer in range(len(tensor_tun)):
        for mapa in range(len(tensor_tun[layer])):
          std_tun_activ[layer][mapa] += (tensor_tun[layer][mapa] - avg_tun_activ[layer][mapa])**2
          
    std_tun_activ = [[(item / len(tun_activ))**(1/2) for item in subl] for subl in avg_tun_activ]

    return std_tun_activ
    
def calc_tun_quality(tun_value):

  tun_quality = [[0 for item in subl] for subl in tun_value[0]]
  for cat in range(len(tun_value)):
    for layer in range(len(tun_value[cat])):
      for map in range(len(tun_value[cat][layer])):
        if tun_quality[layer][map] < abs(tun_value[cat][layer][map]):
          tun_quality[layer][map] = abs(tun_value[cat][layer][map])

  return tun_quality