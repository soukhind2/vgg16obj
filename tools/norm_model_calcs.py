#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 21:56:16 2020

@author: soukhind
"""

import numpy as np
import tensorflow as tf
from keras.layers import Flatten,Dense,Dropout,Input,Activation,Conv2D,MaxPool2D,Lambda
from keras.models import Sequential,Model,load_model
from tensorflow.keras.activations import relu
from keras.utils.generic_utils import get_custom_objects
from tensorflow.keras.callbacks import EarlyStopping
#from vis.utils import utils
from tensorflow.python.ops import nn
from tensorflow import math
import time
from sklearn.metrics import roc_curve,accuracy_score,precision_recall_curve,f1_score
from keras.applications.vgg16 import VGG16
from keras.optimizers import Adam



def gen_div_norm_model():
    
    vgg_model = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape = [224,224,3])
      
    model = Sequential()
    model.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(Lambda(div_norm_2d))
    model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(Lambda(div_norm_2d))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Lambda(div_norm_2d))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Lambda(div_norm_2d))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Lambda(div_norm_2d))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Lambda(div_norm_2d))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Lambda(div_norm_2d))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Lambda(div_norm_2d))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Lambda(div_norm_2d))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Lambda(div_norm_2d))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Lambda(div_norm_2d))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Lambda(div_norm_2d))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Lambda(div_norm_2d))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Lambda(div_norm_2d))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Lambda(div_norm_2d))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Lambda(div_norm_2d))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Lambda(div_norm_2d))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Lambda(div_norm_2d))
    model.add(Flatten())
    model.add(Dense(4096, activation = 'relu',name = 'top_dense1'))
    model.add(Dense(2,activation='softmax',name = 'predictions'))
    
    model.layers[2].set_weights(vgg_model.layers[2].get_weights())
    model.layers[4].set_weights(vgg_model.layers[3].get_weights())
    model.layers[6].set_weights(vgg_model.layers[4].get_weights())
    model.layers[8].set_weights(vgg_model.layers[5].get_weights())
    model.layers[10].set_weights(vgg_model.layers[6].get_weights())
    model.layers[12].set_weights(vgg_model.layers[7].get_weights())
    model.layers[14].set_weights(vgg_model.layers[8].get_weights())
    model.layers[16].set_weights(vgg_model.layers[9].get_weights())
    model.layers[18].set_weights(vgg_model.layers[10].get_weights())
    model.layers[20].set_weights(vgg_model.layers[11].get_weights())
    model.layers[22].set_weights(vgg_model.layers[12].get_weights())
    model.layers[24].set_weights(vgg_model.layers[13].get_weights())
    model.layers[26].set_weights(vgg_model.layers[14].get_weights())
    model.layers[28].set_weights(vgg_model.layers[15].get_weights())
    model.layers[30].set_weights(vgg_model.layers[16].get_weights())
    model.layers[32].set_weights(vgg_model.layers[17].get_weights())
    model.layers[34].set_weights(vgg_model.layers[18].get_weights())
    
    
    
    model.compile(optimizer= Adam(lr=1e-5),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model

def div_norm_2d(x,
                sum_window = [3,3],
                sup_window= [6,6],
                gamma=None,
                beta=None,
                eps=1.0, 
                scope="dn",
                name="dn_out",
                return_mean=False):
  """Applies divisive normalization on CNN feature maps.
    Collect mean and variances on x on a local window across channels. 
    And apply normalization as below:
      x_ = gamma * (x - mean) / sqrt(var + eps) + beta
    Args:
      x: Input tensor, [B, H, W, C].
      sum_window: Summation window size, [H_sum, W_sum].
      sup_window: Suppression window size, [H_sup, W_sup].
      gamma: Scaling parameter.
      beta: Bias parameter.
      eps: Denominator bias.
      return_mean: Whether to also return the computed mean.
    Returns:
      normed: Divisive-normalized variable.
      mean: Mean used for normalization (optional).
    """
  with tf.compat.v1.variable_scope(scope):
    w_sum = tf.ones(sum_window + [1, 1]) / np.prod(np.array(sum_window))
    w_sup = tf.ones(sup_window + [1, 1]) / np.prod(np.array(sum_window))
    x_mean = tf.reduce_mean(x, [3], keepdims=True)
    x_mean = tf.nn.conv2d(x_mean, w_sum, strides=[1, 1, 1, 1], padding='SAME')
    normed = x - x_mean
    x2 = tf.square(normed)
    x2_mean = tf.reduce_mean(x2, [3], keepdims=True)
    x2_mean = tf.nn.conv2d(x2_mean, w_sup, strides=[1, 1, 1, 1], padding='SAME')
    denom = tf.sqrt(x2_mean + eps)
    normed = normed / denom
    if gamma is not None:
      normed *= gamma
    if beta is not None:
      normed += beta
    normed = tf.identity(normed, name=name)
  if return_mean:
    return normed, x_mean
  else:
    return normed



def gen_attnmap(modifier,mask,category,bi,atype):
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
    bi : boolean 
        bidirectionality
        True & False.
    atype: int
        1 = Multiplicative
        2 = Additive
    Returns
    -------
    tensor_attnmap : tensor
        attention map.

    """
    attnmap = []
    #beta = calc_beta(avg_tun_activ)/10
  
    #conv1_1 & conv1_2
    for layer in range(2):
        mapval = np.float32(modifier[category][layer])
        if bi == False:
            mapval[mapval < 0] = 0
        if atype == 1:
          amap = np.ones((224,224,64),dtype='float32') + np.tile(mapval,[224,224,1])* mask[layer]
        elif atype == 2:
          amap = np.tile(mapval,[224,224,1])* mask[layer]
        #amap[amap < 0] = 0
        attnmap.append(amap)
    
    #conv2_1 & conv2_2
    for layer in range(2,4):
        mapval = np.float32(modifier[category][layer])
        if bi == False:
            mapval[mapval < 0] = 0
        if atype == 1:
          amap = np.ones((112,112,128),dtype='float32') + np.tile(mapval,[112,112,1])* mask[layer]
        elif atype == 2:
          amap = np.tile(mapval,[112,112,1])* mask[layer]
        #amap[amap < 0] = 0
        attnmap.append(amap)
    
    #conv3_1 - conv3_3
    for layer in range(4,7):
        mapval = np.float32(modifier[category][layer])
        if bi == False:
            mapval[mapval < 0] = 0
        if atype == 1:
          amap = np.ones((56,56,256),dtype='float32') + np.tile(mapval,[56,56,1])* mask[layer]
        elif atype == 2:
          amap = np.tile(mapval,[56,56,1])* mask[layer]
        #amap[amap < 0] = 0
        attnmap.append(amap)
    
    #conv4_1 - conv4_3
    for layer in range(7,10):
        mapval = np.float32(modifier[category][layer])
        if bi == False:
            mapval[mapval < 0] = 0
        if atype == 1:
          amap = np.ones((28,28,512),dtype='float32') + np.tile(mapval,[28,28,1])* mask[layer]
        elif atype == 2:
          amap = np.tile(mapval,[28,28,1])* mask[layer]
        #amap[amap < 0] = 0
        attnmap.append(amap)
    
    #conv5_1 - conv5_3
    for layer in range(10,13):
        mapval = np.float32(modifier[category][layer])
        if bi == False:
            mapval[mapval < 0] = 0
        if atype ==1:
          amap = np.ones((14,14,512),dtype='float32') + np.tile(mapval,[14,14,1])* mask[layer]
        elif atype == 2:
          amap = np.tile(mapval,[14,14,1])* mask[layer]
        #amap[amap < 0] = 0
        attnmap.append(amap)
    
    tensor_attnmap = []
    for layer in range(len(attnmap)):
      tensor_attnmap.append(tf.convert_to_tensor(attnmap[layer])) 
    
    return tensor_attnmap



def avg_accuracy(data_train,train_labels,
                 data_test,test_labels,
                 modifier,
                 idxpath,
                 category,
                 atstrng,
                 bidir = True,
                 atype = 1):
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

    idxpath : string
        for internal use.
    atstrng : float32
        attention strength.
    bidir : bool, optional
        Bidirectionality. The default is True.
    atype: int
        1 = Multiplicative
        2 = Additive
    Returns
    -------
    t_acc
        Accuracy for each category at each layer.

    """
   

    
    epochs = 30    
    n_layers = 13
    t_acc = np.zeros(n_layers)
    thr = np.zeros(n_layers)
    vgg_model = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape = [224,224,3])
                  
    
    for li in range(n_layers):
        layeridx = 0
        np.save('layeridx.npy',layeridx)
        layermask = np.zeros(13)
        layermask[li] = 1
        tensor_attnmap = gen_attnmap(modifier,layermask*atstrng,category,bidir,atype)
        
        def div_norm_2d_modified(x, mod = tensor_attnmap,
               sum_window = [3,3],
               sup_window= [6,6],
               gamma=None,
               beta=None,
               eps=1.0, 
               scope="dn",
               name="dn_out",
               return_mean=False):
         
            layeridx = np.load('layeridx.npy')
            with tf.compat.v1.variable_scope(scope):
              w_sum = tf.ones(sum_window + [1, 1]) / np.prod(np.array(sum_window))
              w_sup = tf.ones(sup_window + [1, 1]) / np.prod(np.array(sum_window))
              x_mean = tf.reduce_mean(x, [3], keepdims=True)
              x_mean = tf.nn.conv2d(x_mean, w_sum, strides=[1, 1, 1, 1], padding='SAME')
              normed = x - x_mean
              x2 = tf.square(normed)
              x2_mean = tf.reduce_mean(x2, [3], keepdims=True)
              x2_mean = tf.nn.conv2d(x2_mean, w_sup, strides=[1, 1, 1, 1], padding='SAME')
              denom = tf.sqrt(x2_mean + eps)
              normed = normed / denom
              if gamma is not None:
                normed *= gamma
              if beta is not None:
                normed += beta
           
              if layeridx == 13:
                layeridx = 0
              normed *= mod[layeridx]
              #layeridx += 1
              
              np.save('layeridx.npy',layeridx)
              normed = tf.identity(normed, name=name)
            if return_mean:
              return normed, x_mean
            else:
              return normed
    
        model = Sequential()
        model.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
        model.add(Lambda(div_norm_2d_modified))
        layeridx +=1
        np.save('layeridx.npy',layeridx)
        model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
        model.add(Lambda(div_norm_2d_modified))
        layeridx +=1
        np.save('layeridx.npy',layeridx)
        
        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        
        model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Lambda(div_norm_2d_modified))
        layeridx +=1
        np.save('layeridx.npy',layeridx)
        model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Lambda(div_norm_2d_modified))
        layeridx +=1
        np.save('layeridx.npy',layeridx)
        
        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        
        np.save('layeridx.npy',layeridx)
        model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Lambda(div_norm_2d_modified))
        layeridx +=1
        np.save('layeridx.npy',layeridx)
        model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Lambda(div_norm_2d_modified))
        layeridx +=1
        np.save('layeridx.npy',layeridx)
        model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Lambda(div_norm_2d_modified))
        layeridx +=1
        np.save('layeridx.npy',layeridx)
        
        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Lambda(div_norm_2d_modified))
        layeridx +=1
        np.save('layeridx.npy',layeridx)
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Lambda(div_norm_2d_modified))
        layeridx +=1
        np.save('layeridx.npy',layeridx)
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Lambda(div_norm_2d_modified))
        layeridx +=1
        np.save('layeridx.npy',layeridx)
        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        
        
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Lambda(div_norm_2d_modified))
        layeridx +=1
        np.save('layeridx.npy',layeridx)
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Lambda(div_norm_2d_modified))
        layeridx +=1
        np.save('layeridx.npy',layeridx)
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Lambda(div_norm_2d_modified))
        layeridx +=1
        np.save('layeridx.npy',layeridx)
        
        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        
        model.add(Flatten())
        model.add(Dense(4096, activation = 'relu',name = 'top_dense1'))
        model.add(Dense(2,activation='softmax',name = 'predictions'))
        
        model.layers[2].set_weights(vgg_model.layers[2].get_weights())
        model.layers[4].set_weights(vgg_model.layers[3].get_weights())
        model.layers[7].set_weights(vgg_model.layers[5].get_weights())
        model.layers[9].set_weights(vgg_model.layers[6].get_weights())
        model.layers[12].set_weights(vgg_model.layers[8].get_weights())
        model.layers[14].set_weights(vgg_model.layers[9].get_weights())
        model.layers[16].set_weights(vgg_model.layers[10].get_weights())
        model.layers[19].set_weights(vgg_model.layers[12].get_weights())
        model.layers[21].set_weights(vgg_model.layers[13].get_weights())
        model.layers[23].set_weights(vgg_model.layers[14].get_weights())
        model.layers[26].set_weights(vgg_model.layers[16].get_weights())
        model.layers[28].set_weights(vgg_model.layers[17].get_weights())
        model.layers[30].set_weights(vgg_model.layers[18].get_weights())
        
        
        
        
        model.compile(optimizer= Adam(lr=1e-5),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        
        
        history = model.fit(x = data_train,  y = train_labels,
                epochs=epochs,
                batch_size=64,
                verbose = 1, callbacks = [])
    
        out = model.evaluate(data_test, test_labels)
        t_acc[li] = out[1]

    return t_acc


def calc_beta(avg_act):
  beta = [0 for item in avg_act]
  for layer in range(len(avg_act)):
    for item in avg_act[layer]:
      beta[layer] += np.mean(item)
      beta = np.array(beta)
  return beta