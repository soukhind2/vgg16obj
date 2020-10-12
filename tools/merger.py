# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


#%%
import matplotlib.pyplot as plt
from vgg16obj.tools import img_manipulation as im

# Run twice for both correct and incorrect
im.generate_merged('/Users/soukhind/Desktop/ann/data/train/',
                '/Users/soukhind/Desktop/ann/data/merge/merge_data',
                150,correct = False)

#%%

# Run twice for both correct and incorrect
im.sort_merged_imgs('/Users/soukhind/Desktop/ann/data/merge/merge_data/',
          '/Users/soukhind/Desktop/ann/data/merge/merge_train/',
          '/Users/soukhind/Desktop/ann/data/merge/merge_test',
          correct = True)

#%%
from vgg16obj.tools import img_manipulation as im

# Run twice for both correct and incorrect
im.sort_reg_imgs('/Users/soukhind/Desktop/ann/data/merge/merge_reg/',
          '/Users/soukhind/Desktop/ann/data/merge/merge_reg_train/',
          '/Users/soukhind/Desktop/ann/data/merge/merge_reg_test',
          correct = True)