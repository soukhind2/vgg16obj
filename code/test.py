#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 19:45:58 2020

@author: sdas
"""
import numpy as np
import os
import time
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
import math

#%%
model = VGG16(weights='imagenet',
              include_top=False)
#%%
batch_size = 32
datagen = ImageDataGenerator()
train_it = datagen.flow_from_directory('stim/',
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
predict_size_test = int(math.ceil(nb_test_samples / batch_size)) 
#%%
start = time.time()
features_train = model.predict_generator(train_it, predict_size_train) 
print(f'Train Time: {time.time() - start}')

start = time.time()
features_val = model.predict_generator(val_it, predict_size_val) 
print(f'Val Time: {time.time() - start}')

start = time.time()
features_test = model.predict_generator(test_it, predict_size_test) 
print(f'Test Time: {time.time() - start}')

     
np.save('features_train' , features_train)
np.save('features_val', features_val)
np.save('features_test', features_test)
#%%
from keras.layers import Flatten,Dense,Dropout
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
epochs = 30

train_data = np.load('features_train.npy')
#train_data = features_train
train_labels = train_it.classes
train_labels = to_categorical(train_labels, 3)


val_data = np.load('features_val.npy')
#val_data = features_val
val_labels = val_it.classes
val_labels = to_categorical(val_labels, 3)

test_data = np.load('features_test.npy')
#test_data = features_test
test_labels = test_it.classes
test_labels = to_categorical(test_labels, 3)


model = Sequential()

model.add(Flatten(input_shape=train_data.shape[1:])) 
model.add(Dense(4096, activation='relu')) 
model.add(Dropout(0.5)) 
model.add(Dense(1024, activation='relu')) 
model.add(Dense(3, activation='softmax'))

model.compile(optimizer= Adam(lr=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)

history = model.fit(train_data, train_labels,
          epochs=epochs,
          batch_size=batch_size,
          validation_data=(val_data, val_labels),
          verbose = 1, callbacks = [])
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
import itertools
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
categorical_test_labels = pd.DataFrame(test_labels).idxmax(axis=1)
categorical_preds = pd.DataFrame(pred).idxmax(axis=1)
confusion_matrix= confusion_matrix(categorical_test_labels, categorical_preds)

#To get better visual of the confusion matrix:
def plot_confusion_matrix(cm, classes,
   normalize=False,
   title='Confusion matrix',
   cmap=plt.cm.Blues):
 
#Add Normalization Option
 
   if normalize:
     cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
     print('Normalized confusion matrix')
   else:
     print('Confusion matrix, without normalization')
 
# print(cm)
 
   plt.imshow(cm, interpolation='nearest', cmap=cmap)
   plt.title(title)
   plt.colorbar()
   tick_marks = np.arange(len(classes))
   plt.xticks(tick_marks, classes, rotation=45)
   plt.yticks(tick_marks, classes)
 
   fmt = '.2f' if normalize else 'd'
   thresh = cm.max() / 2.
   for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      plt.text(j, i, format(cm[i, j], fmt), horizontalalignment='center', color='white' if cm[i, j] > thresh else 'black')
 
   plt.tight_layout()
   plt.ylabel('True label')
   plt.xlabel('Predicted label') 
                                       
plot_confusion_matrix(confusion_matrix,classes,normalize = False)
#%%
import matplotlib.pyplot as plt
#Graphing our training and validation
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))
plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.ylabel('accuracy') 
plt.xlabel('epoch')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.ylabel('loss') 
plt.xlabel('epoch')
plt.legend()
plt.show()



