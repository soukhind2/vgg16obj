#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 02:39:43 2020

@author: soukhind
"""
import numpy as np
import tensorflow as tf
from keras.layers import Activation
from tensorflow.keras.activations import relu
from keras.utils.generic_utils import get_custom_objects
from keras.callbacks import EarlyStopping
from vis.utils import utils
from tensorflow.python.ops import nn
from tensorflow import math
import time

def gen_attnmap(modifier,mask,category,bi):
    """
    

    Parameters
    ----------
    modifier : list
        modifier to be used to implement attention.
    mask : ndarray
        binary vector to determine which layer to apply attention at. 
        include attention strength by multiplying to it
    category : ndarray
        cateogies .
    bi : bidirectionality
        True & False.

    Returns
    -------
    tensor_attnmap : tensor
        attention map.

    """
    attnmap = []  
    #conv1_1 & conv1_2
    for layer in range(2):
        mapval = np.float32(modifier[category][layer])
        if bi == False:
            mapval[mapval < 0] = 0
        amap = np.ones((224,224,64),dtype='float32') + np.tile(mapval,[224,224,1])* mask[layer]
        amap[amap < 0] = 0
        attnmap.append(amap)
    
    #conv2_1 & conv2_2
    for layer in range(2,4):
        mapval = np.float32(modifier[category][layer])
        if bi == False:
            mapval[mapval < 0] = 0
        amap = np.ones((112,112,128),dtype='float32') + np.tile(mapval,[112,112,1])* mask[layer]
        amap[amap < 0] = 0
        attnmap.append(amap)
    
    #conv3_1 - conv3_3
    for layer in range(4,7):
        mapval = np.float32(modifier[category][layer])
        if bi == False:
            mapval[mapval < 0] = 0
        amap = np.ones((56,56,256),dtype='float32') + np.tile(mapval,[56,56,1])* mask[layer]
        amap[amap < 0] = 0
        attnmap.append(amap)
    
    #conv4_1 - conv4_3
    for layer in range(7,10):
        mapval = np.float32(modifier[category][layer])
        if bi == False:
            mapval[mapval < 0] = 0
        amap = np.ones((28,28,512),dtype='float32') + np.tile(mapval,[28,28,1])* mask[layer]
        amap[amap < 0] = 0
        attnmap.append(amap)
    
    #conv5_1 - conv5_3
    for layer in range(10,13):
        mapval = np.float32(modifier[category][layer])
        if bi == False:
            mapval[mapval < 0] = 0
        amap = np.ones((14,14,512),dtype='float32') + np.tile(mapval,[14,14,1])* mask[layer]
        amap[amap < 0] = 0
        attnmap.append(amap)
    
    tensor_attnmap = []
    for layer in range(len(attnmap)):
      tensor_attnmap.append(tf.convert_to_tensor(attnmap[layer])) 
    
    return tensor_attnmap



def avg_accuracy(data_train,train_labels,
                 data_test,test_labels,
                 categories,
                 modifier,
                 model,top_model,idxpath,
                 atstrng,
                 bidir = True):
    """
    

    Parameters
    ----------
    data_train : ndarray
        Training data.
    train_labels : categorical
        Training labels.
    data_test : ndarray
        Testing data.
    test_labels : categorical
        Testing labels.
    categories : ndarray
        Names of each category.
    modifier : list
        modifier to be used to implement attention.
    model : keras model
        base model.
    top_model : keras model
        top model.
    idxpath : string
        for internal use.
    atstrng : float32
        attention strength.
    bidir : bool, optional
        Bidirectionality. The default is True.

    Returns
    -------
    t_acc
        Accuracy for each category at each layer.

    """
    
    ncats = len(categories)
    n_layers = 13
    t_acc = np.zeros((ncats,n_layers))
    epochs = 30
    
    for cat in range(ncats):
        print('Category of interest: ', categories[cat])
        train_it = np.concatenate((data_train[cat],data_train[cat + 6]))
        test_it = np.concatenate((data_test[cat],data_test[cat + 6]))
        print(train_it.shape,test_it.shape)
    
    
        for li in range(n_layers):
            beta = [20,100,150,150,240,240,150,150,80,20,20,10,1] #multiplicative type
            layermask = np.zeros(13)
            layermask[li] = 1
            tensor_attnmap = gen_attnmap(modifier,layermask*atstrng,cat,bidir)
                    

            def attnrelu(x,map = tensor_attnmap):
                layeridx = np.load(idxpath)
                if layeridx == 13:
                    layeridx = 0
                activations = math.multiply(x,map[layeridx])
                activations = nn.relu(activations)
                layeridx += 1
                np.save(idxpath,layeridx)
                return activations
        
            get_custom_objects().update({'attnrelu': Activation(attnrelu)})
        
            for layer in model.layers:
                if(hasattr(layer,'activation')):
                    layer.activation = attnrelu
        
            utils.apply_modifications(model)
            model.compile()
        
            start = time.time()
            f_train = model.predict(train_it) 
            print(f'Train Time: {time.time() - start}')
        
        
            start = time.time()
            f_test = model.predict(test_it)
            print(f'Test Time: {time.time() - start}')
            es = EarlyStopping(monitor='loss', mode='min', verbose=1)
      
            history = top_model.fit(x = f_train,  y = train_labels,
                    epochs=epochs,
                    batch_size=64,
                    verbose = 0, callbacks = [es])
        
            out = top_model.evaluate(f_test, test_labels)
            t_acc[cat,li] = out[1]
    return t_acc


def get_acc(data_train,train_labels,
                 data_test,test_labels,
                 categories,
                 modifier,
                 model,top_model,idxpath,
                 atstrng,
                 bidir = True):

    x = avg_accuracy(data_train,train_labels,
                 data_test,test_labels,
                 categories,
                 modifier,
                 model,top_model,idxpath,
                 atstrng,
                 bidir = True)
    return x