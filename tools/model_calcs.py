#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 02:39:43 2020

@author: soukhind
"""
import numpy as np
import tensorflow as tf
from keras.layers import Activation
from keras.activations import relu
from keras.utils import get_custom_objects
from keras.callbacks import EarlyStopping
from vis.utils import utils
from tensorflow.python.ops import nn
from tensorflow import math
import time
from sklearn.metrics import roc_curve,accuracy_score,precision_recall_curve,f1_score
import keras
from keras.utils import custom_object_scope
from sklearn import svm
from sklearn.model_selection import train_test_split,KFold
import gc 

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
        mapval = np.float32(modifier[category][layer]).reshape(1,1,64)
        mapval = tf.constant(mapval,tf.float32)
        if bi == False:
            mapval[mapval < 0] = 0
        if atype == 1:
          amap = tf.ones((224,224,64),dtype='float32') + tf.tile(mapval,[224,224,1])* mask[layer]
        elif atype == 2:
          amap = tf.tile(mapval,[224,224,1])* mask[layer]
        #amap[amap < 0] = 0
        attnmap.append(amap)

    #conv2_1 & conv2_2
    for layer in range(2,4):
        mapval = np.float32(modifier[category][layer]).reshape(1,1,128)
        if bi == False:
            mapval[mapval < 0] = 0
        if atype == 1:
          amap = tf.ones((112,112,128),dtype='float32') + tf.tile(mapval,[112,112,1])* mask[layer]
        elif atype == 2:
          amap = tf.tile(mapval,[112,112,1])* mask[layer]
        #amap[amap < 0] = 0
        attnmap.append(amap)

    #conv3_1 - conv3_3
    for layer in range(4,7):
        mapval = np.float32(modifier[category][layer]).reshape(1,1,256)
        if bi == False:
            mapval[mapval < 0] = 0
        if atype == 1:
          amap = tf.ones((56,56,256),dtype='float32') + tf.tile(mapval,[56,56,1])* mask[layer]
        elif atype == 2:
          amap = tf.tile(mapval,[56,56,1])* mask[layer]
        #amap[amap < 0] = 0
        attnmap.append(amap)

    #conv4_1 - conv4_3
    for layer in range(7,10):
        mapval = np.float32(modifier[category][layer]).reshape(1,1,512)
        if bi == False:
            mapval[mapval < 0] = 0
        if atype == 1:
          amap = tf.ones((28,28,512),dtype='float32') + tf.tile(mapval,[28,28,1])* mask[layer]
        elif atype == 2:
          amap = tf.tile(mapval,[28,28,1])* mask[layer]
        #amap[amap < 0] = 0
        attnmap.append(amap)

    #conv5_1 - conv5_3
    for layer in range(10,13):
        mapval = np.float32(modifier[category][layer]).reshape(1,1,512)
        if bi == False:
            mapval[mapval < 0] = 0
        if atype ==1:
          amap = tf.ones((14,14,512),dtype='float32') + tf.tile(mapval,[14,14,1])* mask[layer]
        elif atype == 2:
          amap = tf.tile(mapval,[14,14,1])* mask[layer]
        #amap[amap < 0] = 0
        attnmap.append(amap)

    tensor_attnmap = []
    for layer in range(len(attnmap)):
      tensor_attnmap.append(tf.convert_to_tensor(attnmap[layer]))

    return tensor_attnmap





def avg_accuracy(data_train,train_labels,
                 data_test,test_labels,
                 modifier,
                 model,top_model,idxpath,
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
    atype: int
        1 = Multiplicative
        2 = Additive
    Returns
    -------
    t_acc
        Accuracy for each category at each layer.

    """
    folds = 5
    epochs = 30
    n_layers = 13
    t_acc = np.zeros((n_layers,folds))
    thr = np.zeros(n_layers)
    es = EarlyStopping(monitor='loss', mode='min', verbose=1)
    for li in range(n_layers):
        layermask = np.zeros(13)
        layermask[li] = 1
        tensor_attnmap = gen_attnmap(modifier,layermask*atstrng,category,bidir,atype)

        @keras.utils.register_keras_serializable(package="my_package", name="attnrelu")
        def attnrelu(x,map = tensor_attnmap,atype = atype):
            layeridx = np.load(idxpath)
            if layeridx == 13:
                layeridx = 0
            if atype == 1:
              x = nn.relu(x)
              activations = math.multiply(x,map[layeridx])
            if atype == 2:
              activations = math.add(x,map[layeridx])
              activations = nn.relu(activations)
            layeridx += 1
            np.save(idxpath,layeridx)
            return activations

        get_custom_objects().update({'attnrelu': Activation(attnrelu)})
        with custom_object_scope({'attnrelu': attnrelu}):

          for layer in model.layers:
              if(hasattr(layer,'activation')):
                  layer.activation = attnrelu

          del tensor_attnmap
          #utils.apply_modifications(model)
          model.compile()

          f_train = model.predict(data_train)

          kf = KFold(n_splits=folds,shuffle = True)
          for i, (_,testidx) in enumerate(kf.split(data_test)):

            f_test = model.predict(data_test[testidx])
            test_targets = test_labels[testidx]


            history = top_model.fit(x = f_train,  y = train_labels,
                    epochs=epochs,
                    batch_size=32,
                    verbose = 0, callbacks = [])

            out = top_model.evaluate(f_test, test_targets)
            t_acc[li,i] = out[1]
    del model, top_model
    gc.collect()
    #return [t_acc]
    np.save('temp',t_acc)


def calc_beta(avg_act):
  beta = [0 for item in avg_act]
  for layer in range(len(avg_act)):
    for item in avg_act[layer]:
      beta[layer] += np.mean(item)
      beta = np.array(beta)
  return beta

  from sklearn.metrics import accuracy_score


def run_SVM(A,B):

  X = np.concatenate((A,B),0)
  y = np.concatenate((np.zeros(len(A)),np.ones(len(B))),0)

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,shuffle=True)

  clf = svm.SVC(kernel='linear', C=1)
  clf.fit(X_train, y_train)

  y_pred = clf.predict(X_test)

  acc = accuracy_score(y_test, y_pred)

  return acc