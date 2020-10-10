# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


#%%
import matplotlib.pyplot as plt
generate_merged('/Users/soukhind/Desktop/ann/data/train/',
                '/Users/soukhind/Desktop/ann/data/merge/merge_data',
                150,correct = False)

#%%


sort_imgs('/Users/soukhind/Desktop/ann/data/merge/merge_data/',
          '/Users/soukhind/Desktop/ann/data/merge/merge_train/',
          '/Users/soukhind/Desktop/ann/data/merge/merge_test',
          correct = False)