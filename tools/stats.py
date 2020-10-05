#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 02:28:26 2020

@author: soukhind
"""
from scipy.stats import pearsonr
import numpy as np

def calc_corrcoef(data1,data2,layer,map):
  X1 = np.zeros(len(data1))
  #Load data for calculation
  for i in range(len(data1)):
      X1[i] = data1[i][layer][map]
  X2 = np.zeros(len(data2))
  #Load data for calculation
  for i in range(len(data1)):
      X2[i] = data2[i][layer][map]

  corr,_ = pearsonr(X1,X2)
  return np.round(corr,3)

def calc_all_corrcoeff(data1,data2):
  
  if len(data1) != len(data2) or len(data1[0]) != len(data2[0]):
    raise ValueError('Sizes do not match')
    
  X1 = np.zeros(len(data1))
  X2 = np.zeros(len(data2))
  corr = [[0 for item in subl] for subl in data1[0]]
  #Load data into format
  for layer in range(len(data1[0])):
    for map in range(len(data1[0][layer])):
      for cat in range(len(data1)):
        X1[cat] = data1[cat][layer][map]
        X2[cat] = data2[cat][layer][map]
      corr[layer][map],_ = pearsonr(X1,X2)
  return corr
