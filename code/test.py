#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 19:45:58 2020

@author: sdas
"""
import numpy as np
import os
import time

from keras.applications.vgg16 import VGG16,preprocess_input
from keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array
from keras.layers import Flatten,Dense,Dropout
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping


import math
import itertools
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

from tools import plot_tools as pt
#%%
model = VGG16(weights='imagenet',
              include_top=False)
#%%
'''
batch_size = 32
datagen = ImageDataGenerator(rescale = 1./255,  
          zoom_range = 0.1,
          width_shift_range = 0.2, 
          height_shift_range = 0.2,
          horizontal_flip = True,
          fill_mode ='nearest')
train_it = datagen.flow_from_directory('train/',
                                       batch_size = batch_size, 
                                       target_size = (224, 224),
                                       class_mode = 'categorical',
                                       )

val_it = datagen.flow_from_directory('val/',
                                       batch_size = batch_size, 
                                       target_size = (224, 224),
                                       class_mode = 'categorical',
                                       )

test_it = datagen.flow_from_directory('test/',
                                       batch_size = batch_size, 
                                       target_size = (224, 224),
                                       class_mode = 'categorical',
                                       )

num_classes = 3
nb_train_samples = len(train_it.filenames) 
nb_val_samples = len(val_it.filenames) 
nb_test_samples = len(test_it.filenames) 



predict_size_train = int(math.ceil(nb_train_samples / batch_size)) 
predict_size_val = int(math.ceil(nb_val_samples / batch_size)) 
predict_size_test = int(math.ceil(nb_test_samples / batch_size))'''
#%%



def convertimgs(path,data) :
    for dirName, subdir, files in os.walk(path):
        for filename in sorted(files):
            if filename == '.DS_Store':
                continue
            ds = load_img(path +'/' + filename,target_size = (224,224))
            im = img_to_array(ds)
            #im = im.reshape((1, im.shape[0], im.shape[1], im.shape[2]))
            im = preprocess_input(im)
            data.append(im) 
    return data


train_it = []
train_it = convertimgs('data/train/faces',train_it)
train_it = convertimgs('data/train/Scenes',train_it)
train_it = convertimgs('data/train/objects',train_it)

val_it = []
val_it = convertimgs('data/val/faces',val_it)
val_it = convertimgs('data/val/scenes',val_it)
val_it = convertimgs('data/val/objects',val_it)

test_it = []
test_it = convertimgs('data/test/faces',test_it)
test_it = convertimgs('data/test/scenes',test_it)
test_it = convertimgs('data/test/objects',test_it)

train_it = np.array(train_it)
val_it = np.array(val_it)
test_it = np.array(test_it)
print(train_it.shape,val_it.shape,test_it.shape)
#%%
start = time.time()
features_train = model.predict(train_it) 
print(f'Train Time: {time.time() - start}')

start = time.time()
features_val = model.predict(val_it) 
print(f'Val Time: {time.time() - start}')

start = time.time()
features_test = model.predict(test_it) 
print(f'Test Time: {time.time() - start}')

     
np.save('features_train' , features_train)
np.save('features_val', features_val)
np.save('features_test', features_test)
#%%

epochs = 30

train_data = np.load('features_train.npy')
#train_data = features_train
train_labels = [0] * 500 + [1]*500 + [2] * 500
train_labels = to_categorical(train_labels, 3)


val_data = np.load('features_val.npy')
#val_data = features_val
val_labels = [0] * 50 + [1]*50 + [2] * 50
val_labels = to_categorical(val_labels, 3)

test_data = np.load('features_test.npy')
#test_data = features_test
test_labels = [0] * 50 + [1]*50 + [2] * 50
test_labels = to_categorical(test_labels, 3)


model = Sequential()

model.add(Flatten(input_shape=train_data.shape[1:])) 
model.add(Dense(4096, activation='relu')) 
model.add(Dropout(0.5)) 
model.add(Dense(1024, activation='relu')) 
model.add(Dense(3, activation='softmax'))

model.compile(optimizer= Adam(lr=1e-5),
              loss='binary_crossentropy',
              metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)

history = model.fit(train_data, train_labels,
          epochs=epochs,
          batch_size=64,
          validation_data=(val_data, val_labels),
          verbose = 1, callbacks = [es])
model.save_weights('top_weights',overwrite = True)

#%%
out = model.evaluate(test_data,test_labels)
print(out)
#%%
pred = np.round(model.predict(test_data),0)
print('rounded test labels',pred)

#%%
from sklearn import metrics
classes = ['faces','object','scene']
metric = metrics.classification_report(test_labels,pred,target_names = classes)
print(metric)

#%%

categorical_test_labels = pd.DataFrame(test_labels).idxmax(axis=1)
categorical_preds = pd.DataFrame(pred).idxmax(axis=1)
cm = confusion_matrix(categorical_test_labels, categorical_preds)

pt.plot_confusion_matrix(cm,classes,normalize = False)
#%%
pt.plot_metrics(history)
