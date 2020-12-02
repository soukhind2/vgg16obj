#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 20:13:20 2020

@author: soukhind
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 03:53:33 2020

@author: soukhind
"""
import numpy as np
import pandas as pd
import os
#os.environ["KERAS_BACKEND"] = "plaidml.keras.backend" 
import time
import math
import matplotlib.pyplot as plt
import pickle
import cv2

from keras.applications.vgg16 import VGG16,preprocess_input
from keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array
from keras.layers import Flatten,Dense,Dropout,Input
from keras.models import Sequential,Model,load_model
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.utils.vis_utils import plot_model

from sklearn import metrics
from sklearn.metrics import confusion_matrix
import itertools

from vgg16obj.tools import tuning_calcs as tc
from vgg16obj.tools import gradient_calcs as gc
from vgg16obj.tools import stats as st

#import tensorflow as tf
#from tensorflow import math
import seaborn as sns



#%%
def noisy(image):
  row,col,ch= image.shape
  mean = 0
  var = 1
  sigma = var**0.5
  gauss = np.random.normal(mean,sigma,(row,col,ch))
  gauss = gauss.reshape(row,col,ch)
  noisy = image + gauss
  return noisy

def convertimgs(path,noise = False) :
    data = []
    for dirName, subdir, files in os.walk(path):
        for filename in sorted(files):
            if filename == '.DS_Store':
                continue
            ds = load_img(path +'/' + filename,target_size = (224,224))
            im = img_to_array(ds)
            im /= 255.
            #im = im.reshape((1, im.shape[0], im.shape[1], im.shape[2]))
            #im = preprocess_input(im)
            if noise:
              im = noisy(im)
            data.append(im) 
    return data

#%%
# Merged images load
data_train = [[]  for i in range(12)]
data_train[0] = convertimgs('/Users/soukhind/Desktop/ann/data/merge/merge_train/Correct/Male',noise = False)  #75
data_train[1] = convertimgs('/Users/soukhind/Desktop/ann/data/merge/merge_train/Correct/Female',noise = False) #75
data_train[2] = convertimgs('/Users/soukhind/Desktop/ann/data/merge/merge_train/Correct/Manmade',noise = False) #75
data_train[3] = convertimgs('/Users/soukhind/Desktop/ann/data/merge/merge_train/Correct/Natural',noise = False) #75
data_train[4] = convertimgs('/Users/soukhind/Desktop/ann/data/merge/merge_train/Correct/Powered',noise = False) #75
data_train[5] = convertimgs('/Users/soukhind/Desktop/ann/data/merge/merge_train/Correct/Nonpowered',noise = False) #75

data_train[6] = convertimgs('/Users/soukhind/Desktop/ann/data/merge/merge_train/Incorrect/Male',noise = False) #75
data_train[7] = convertimgs('/Users/soukhind/Desktop/ann/data/merge/merge_train/Incorrect/Female',noise = False) #75
data_train[8] = convertimgs('/Users/soukhind/Desktop/ann/data/merge/merge_train/Incorrect/Manmade',noise = False) #75
data_train[9] = convertimgs('/Users/soukhind/Desktop/ann/data/merge/merge_train/Incorrect/Natural',noise = False) #75
data_train[10] = convertimgs('/Users/soukhind/Desktop/ann/data/merge/merge_train/Incorrect/Powered',noise = False) #75
data_train[11] = convertimgs('/Users/soukhind/Desktop/ann/data/merge/merge_train/Incorrect/Nonpowered',noise = False) #75
data_train = np.array(data_train)
 
data_test = [[]  for i in range(12)]
data_test[0] = convertimgs('/Users/soukhind/Desktop/ann/data/merge/merge_test/Correct/Male',noise = False)  #75
data_test[1] = convertimgs('/Users/soukhind/Desktop/ann/data/merge/merge_test/Correct/Female/',noise = False) #75
data_test[2] = convertimgs('/Users/soukhind/Desktop/ann/data/merge/merge_test/Correct/Manmade/',noise = False) #75
data_test[3] = convertimgs('/Users/soukhind/Desktop/ann/data/merge/merge_test/Correct/Natural/',noise = False) #75
data_test[4] = convertimgs('/Users/soukhind/Desktop/ann/data/merge/merge_test/Correct/Powered',noise = False) #75
data_test[5] = convertimgs('/Users/soukhind/Desktop/ann/data/merge/merge_test/Correct/Nonpowered',noise = False) #75

data_test[6] = convertimgs('/Users/soukhind/Desktop/ann/data/merge/merge_test/Incorrect/Male',noise = False) #75
data_test[7] = convertimgs('/Users/soukhind/Desktop/ann/data/merge/merge_test/Incorrect/Female',noise = False) #75
data_test[8] = convertimgs('/Users/soukhind/Desktop/ann/data/merge/merge_test/Incorrect/Manmade',noise = False) #75
data_test[9] = convertimgs('/Users/soukhind/Desktop/ann/data/merge/merge_test/Incorrect/Natural',noise = False) #75
data_test[10] = convertimgs('/Users/soukhind/Desktop/ann/data/merge/merge_test/Incorrect/Powered',noise = False) #75
data_test[11] = convertimgs('/Users/soukhind/Desktop/ann/data/merge/merge_test/Incorrect/Nonpowered',noise = False) #75
data_test = np.array(data_test)


print(data_train.shape,data_test.shape)

plt.axis('off')

# Regular Images load

reg_train = [[]  for i in range(12)]
reg_train[0] = convertimgs('/Users/soukhind/Desktop/ann/data/merge/merge_reg_train/Correct/Male',noise = False)  # 75
reg_train[1] = convertimgs('/Users/soukhind/Desktop/ann/data/merge/merge_reg_train/Correct/Female',noise = False) # 75
reg_train[2] = convertimgs('/Users/soukhind/Desktop/ann/data/merge/merge_reg_train/Correct/Manmade',noise = False) # 75
reg_train[3] = convertimgs('/Users/soukhind/Desktop/ann/data/merge/merge_reg_train/Correct/Natural',noise = False) # 75
reg_train[4] = convertimgs('/Users/soukhind/Desktop/ann/data/merge/merge_reg_train/Correct/Powered',noise = False) # 75
reg_train[5] = convertimgs('/Users/soukhind/Desktop/ann/data/merge/merge_reg_train/Correct/Nonpowered',noise = False) # 75

reg_train[6] = convertimgs('/Users/soukhind/Desktop/ann/data/merge/merge_reg_train/Incorrect/Male',noise = False)  # 75
reg_train[7] = convertimgs('/Users/soukhind/Desktop/ann/data/merge/merge_reg_train/Incorrect/Female',noise = False) # 75
reg_train[8] = convertimgs('/Users/soukhind/Desktop/ann/data/merge/merge_reg_train/Incorrect/Manmade',noise = False) # 75
reg_train[9] = convertimgs('/Users/soukhind/Desktop/ann/data/merge/merge_reg_train/Incorrect/Natural',noise = False) # 75
reg_train[10] = convertimgs('/Users/soukhind/Desktop/ann/data/merge/merge_reg_train/Incorrect/Powered',noise = False) # 75
reg_train[11] = convertimgs('/Users/soukhind/Desktop/ann/data/merge/merge_reg_train/Incorrect/Nonpowered',noise = False) # 75
reg_train = np.array(reg_train)


reg_test = [[]  for i in range(12)]
reg_test[0] = convertimgs('/Users/soukhind/Desktop/ann/data/merge/merge_reg_test/Correct/Male',noise = False)  #75
reg_test[1] = convertimgs('/Users/soukhind/Desktop/ann/data/merge/merge_reg_test/Correct/Female/',noise = False) #75
reg_test[2] = convertimgs('/Users/soukhind/Desktop/ann/data/merge/merge_reg_test/Correct/Manmade/',noise = False) #75
reg_test[3] = convertimgs('/Users/soukhind/Desktop/ann/data/merge/merge_reg_test/Correct/Natural/',noise = False) #75
reg_test[4] = convertimgs('/Users/soukhind/Desktop/ann/data/merge/merge_reg_test/Correct/Powered',noise = False) #75
reg_test[5] = convertimgs('/Users/soukhind/Desktop/ann/data/merge/merge_reg_test/Correct/Nonpowered',noise = False) #75

reg_test[6] = convertimgs('/Users/soukhind/Desktop/ann/data/merge/merge_reg_test/Incorrect/Male',noise = False) #75
reg_test[7] = convertimgs('/Users/soukhind/Desktop/ann/data/merge/merge_reg_test/Incorrect/Female',noise = False) #75
reg_test[8] = convertimgs('/Users/soukhind/Desktop/ann/data/merge/merge_reg_test/Incorrect/Manmade',noise = False) #75
reg_test[9] = convertimgs('/Users/soukhind/Desktop/ann/data/merge/merge_reg_test/Incorrect/Natural',noise = False) #75
reg_test[10] = convertimgs('/Users/soukhind/Desktop/ann/data/merge/merge_reg_test/Incorrect/Powered',noise = False) #75
reg_test[11] = convertimgs('/Users/soukhind/Desktop/ann/data/merge/merge_reg_test/Incorrect/Nonpowered',noise = False) #75
reg_test = np.array(reg_test)

print(reg_train.shape,reg_test.shape)
#%%

from keras_vggface.vggface import VGGFace
vggface = VGGFace(model='vgg16',include_top = 'False',input_shape=(224, 224, 3))


#%%
# Calling VGG16 without the top FC layers with weights loaded from imagenet 
model = VGG16(weights=None,
              include_top=False,input_shape = [224,224,3])
model.set_weights(vggface.get_weights())

#%%
#plot_model(model,show_shapes=True,expand_nested=True)
#model.save_weights('vgg_w',save_format='h5')

categories = ['Male','Female','Manmade','Natural','Powered','Nonpowered']
interest = 0
print('Category of interest: ', categories[interest])
train_it = np.concatenate((reg_train[interest],reg_train[interest + 6]))
test_it = np.concatenate((reg_test[interest],reg_test[interest + 6]))
print(train_it.shape,test_it.shape)

start = time.time()
features_train = model.predict(train_it) 
print(f'Train Time: {time.time() - start}')

start = time.time()
features_test = model.predict(test_it) 
print(f'Test Time: {time.time() - start}')
epochs = 30

ntrain = 80
train_labels = to_categorical([0] * ntrain + [1]*ntrain)


ntest = 40
test_labels = to_categorical([0] * ntest + [1]*ntest) 

losses = 'binary_crossentropy'

# Initialize the top model with FC layers. These are the only ones to be trained
top_model = Sequential()
top_model.add(Flatten(input_shape=features_train.shape[1:])) 
top_model.add(Dense(4096, activation='relu',name = 'top_dense1')) 
top_model.add(Dense(2, activation='softmax',name = 'predictions'))

top_model.compile(optimizer= Adam(lr=1e-5),
              loss=losses,
              metrics=['accuracy'])
top_model.summary()

es = EarlyStopping(monitor='loss', mode='min', verbose=1)


#%%

#%%
categories = ['Male','Female','Manmade','Natural','Powered','Nonpowered']

acc = np.zeros((2,6))
for imtype in range(2):
  for cat in range(6):

    print('Category of interest: ', categories[cat])
    train_it = np.concatenate((reg_train[cat],reg_train[cat + 6])) # Train on regular data
    if imtype == 0: # Regular
      test_it = np.concatenate((reg_test[cat],reg_test[cat + 6])) # Test on merged data
    else:
      test_it = np.concatenate((data_test[cat],data_test[cat + 6])) # Test on merged data
    print(train_it.shape,test_it.shape)

    
    start = time.time()
    train_data = model.predict(train_it) 
    print(f'Train Time: {time.time() - start}')

    start = time.time()
    test_data = model.predict(test_it) 
    print(f'Test Time: {time.time() - start}')
    
    history = top_model.fit(x = train_data,  y = train_labels,
            epochs=epochs,
            batch_size=64,
            verbose = 1, callbacks = [es])

    out = top_model.evaluate(test_data, test_labels)
    acc[imtype,cat] = out[1]
    print(out)
#%%
data_plot = np.vstack([[0.58,0.66,0.63,0.57,0.7,0.61],acc1[1],acc2[1]]) # Accuracy scores obtained from regular model on merge set
pd_data = pd.DataFrame({
                        'Male':data_plot[:,0],'Female':data_plot[:,1],
                        'Manmade':data_plot[:,2],'Natural':data_plot[:,3],
                        'Powered':data_plot[:,4],'Nonpowered':data_plot[:,5]})

sns.set(style="white",rc={"lines.linewidth": 2,'lines.markersize': 15})
plt.figure(figsize = (8,8))
ax = sns.lineplot(data = pd_data,marker = 'o')
ax.set_xticks([0,1])
ax.set_xlabel('Model type',size = 20)
ax.set_title('Performance of different models on merged images',size = 20)
ax.set_xticklabels(['Regular','Normalised'],size = 15)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.set_ylabel('Binary Classification Acc.',size = 20)
#%%
categories = ['Male','Female','Manmade','Natural','Powered','Nonpowered']
calc_tun_activ = 1

if calc_tun_activ:
  for cat in range(6): 

    print('Category of interest: ', categories[cat])
    train_it = reg_train[cat]

    layer_names = [layer.name for layer in model.layers if 'conv' in layer.name]
    layer_outputs = [layer.output for layer in model.layers if layer.name in layer_names]
    activation_model = Model(inputs=model.input, outputs=layer_outputs)


    start = time.time()
    tun_activ = tc.calc_tun(train_it,activation_model,layer_names)
    print(f'Elapsed: {time.time() - start}s')

    with open('tuning_values_' + str(cat), 'wb') as fp:
      pickle.dump(tun_activ, fp)


#%%
tun_activ = []
for interest in range(6): 
  with open ('tuning_values_' + str(interest), 'rb') as fp:
      tun_activ.extend(pickle.load(fp))
len(tun_activ)

#%%

ncats = 6
# to fish out each category tun_activations
labels = np.array([0] * 40 + [1] * 40 + [2] * 40 + [3] * 40 + [4] * 40 + [5] * 40)
cat_tun = [[[] for j in range(len(tun_activ[0]))] for i in range(ncats)]

avg_tun_activ = tc.calc_avg(tun_activ) #average tuning activity for each map
std_tun_activ = tc.calc_std(tun_activ,avg_tun_activ) #std tuning activity for each map

for i in range(ncats):
  #Calculating average activity of each
  #feature map in response to images of respective category, 
  #with the mean activity under all image categories subtracted from it
  idx = list(np.where(labels == i))
  cat_tun[i] = tc.calc_avg([tun_activ[i] for i in idx[0]])

# Vector of tuning values for each obj cataegory
fc = [[[0 for item in subl] for subl in cat_tun[0]] for i in range(ncats)]
#fc has length of ncats x nlayers x no of maps in each layer
for cat in range(ncats):
  for layer in range(len(cat_tun[cat])):
    for map in range(len(cat_tun[cat][layer])):
      if std_tun_activ[layer][map] == 0:
        continue
      fc[cat][layer][map] = (cat_tun[cat][layer][map] - 
                                avg_tun_activ[layer][map])/std_tun_activ[layer][map]
      

#%%
tun_quality = tc.calc_tun_quality(fc)
sns.set(style="whitegrid")
ax = sns.boxplot(data = tun_quality,palette='cool')
ax.set_xlabel('Layer')
ax.set_xticklabels(np.arange(1,14))
ax.set_ylabel('Tuning Quality')
ax.set_ylim([0,50])
#%%
#@title Multiplicative and bidirectionality corrected

#@title Multiplicative and bidirectionality corrected

from vgg16obj.tools import model_calcs as mc
import gc
layeridx = 0
np.save('layeridx',layeridx)
#atstrng = 10.5 #0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10
for atstrng in np.arange(0,5,1): #Calling for different attn strengths
  grand_acc = [[] for i in range(6)] 
  for cat in range(6): # Iterating over 6 categories
    tf.keras.backend.clear_session()
    print('Category of interest: ', categories[cat])
    train_it = np.concatenate((reg_train[cat],reg_train[cat + 6])) # Training on regular data
    test_it = np.concatenate((data_test[cat],data_test[cat + 6])) # Testing on merged data
    print(train_it.shape,test_it.shape)
    model = VGG16(weights='imagenet',
                        include_top=False,input_shape = [224,224,3])
          
    top_model = Sequential()
    top_model.add(Flatten(input_shape=features_train.shape[1:])) 
    top_model.add(Dense(4096, activation='relu',name = 'top_dense1')) 
    top_model.add(Dense(2, activation='sigmoid',name = 'predictions'))
    top_model.compile(optimizer= Adam(lr=1e-5),
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    
    # Read avg_accuracy's definition to learn about the parameters
    grand_acc[cat] = mc.avg_accuracy(train_it,
                              train_labels,
                              test_it,
                              test_labels,
                              fc,
                              model,
                              top_model,
                              '/content/layeridx.npy',
                              cat,
                              atstrng,
                              bidir = True,
                              atype = 1 )
  grand_acc = np.array(grand_acc)
  grand_acc.reshape([1,6,13])
  # Use tun_accuracy_multi for multiplicative mode (atype = 1) and
  # use tun_accuracy_bias for bias mode (atype = 2)
  np.save('tun_accuracy_multi' + str(atstrng),grand_acc)



#%%

