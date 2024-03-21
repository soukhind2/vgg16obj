#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 19:34:44 2020

@author: sdas
"""

import matplotlib.pyplot as plt
import numpy as np
import itertools

#Graphing the training and validation
def plot_metrics(history,save = False):
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
    if save:
        plt.savefig("metrics1.png",dpi = 600,bbox_inches = 'tight')

    plt.figure()
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.ylabel('loss') 
    plt.xlabel('epoch')
    plt.legend()
    plt.show()
    if save:
        plt.savefig("metrics2.png",dpi = 600,bbox_inches = 'tight')
    
    

#To get better visual of the confusion matrix:
def plot_confusion_matrix(cm, classes,
   normalize=False,
   title='Confusion matrix',
   cmap=plt.cm.Blues,
   save = False):
 
    #Add Normalization Option
 
   if normalize:
     cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
     print('Normalized confusion matrix')
   else:
     print('Confusion matrix, without normalization')
 
 
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
   if save:
        plt.savefig("metrics2.png",dpi = 600,bbox_inches = 'tight')
        
#To visualize and plot the feature maps for reference
def plot_feat_maps(layer_names,intermediate_activations,
                   images_per_row = 8,
                   max_images = 8):
    
    images_per_row = images_per_row
    max_images = max_images
    # Now let's display our feature maps
    for layer_name, layer_activation in zip(layer_names, intermediate_activations):
        # This is the number of features in the feature map
        n_features = layer_activation.shape[-1]
        n_features = min(n_features, max_images)
    
        # The feature map has shape (1, size, size, n_features)
        size = layer_activation.shape[1]
    
        # We will tile the activation channels in this matrix
        n_cols = n_features // images_per_row
        display_grid = np.zeros((size * n_cols, images_per_row * size))
    
        # We'll tile each filter into this big horizontal grid
        for col in range(n_cols):
            for row in range(images_per_row):
                channel_image = layer_activation[0,
                                                 :, :,
                                                 col * images_per_row + row]
                # Post-process the feature to make it visually palatable
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size : (col + 1) * size,
                             row * size : (row + 1) * size] = channel_image
    
        # Display the grid
        scale = 2. / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.axis('off')
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
        
    plt.show()
               
# To plot  tuning curves of the categories
def plot_curves(data,layers,maps,figsize = (15,4)):
  if len(layers) != len(maps):
    raise ValueError('Sizes of layers and maps do not match')
  
  plot_data = np.zeros((len(data),len(layers)))
  #Load data into plotting format
  for i in range(len(data)):
    for iter in range(len(layers)):
      plot_data[i,iter] = data[i][layers[iter]][maps[iter]]

  plt.style.use('default')
  fig = plt.figure(figsize = figsize)
  ax = fig.add_subplot(111)    # The big subplot

  # Turn off axis lines and ticks of the big subplot
  ax.spines['top'].set_color('none')
  ax.spines['bottom'].set_color('none')
  ax.spines['left'].set_color('none')
  ax.spines['right'].set_color('none')
  ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
  
  for p in range(len(layers)):
    ax1 = fig.add_subplot(1,len(layers),p+1)
    ax1.plot(plot_data[:,p])
    ax1.set_title('Layer: ' + str(layers[p]) + ', Map: ' + str(maps[p]))
    ax1.set_xticklabels(np.arange(0,7))
  ax.set_ylabel('Tuning Value',size = 15)
  ax.set_xlabel('Object Category Number',size = 15)
  return fig 

    
from vgg16obj.tools import stats
def plot_both_curves(data1,data2,layers,maps,corrcoef = False,figsize = (20,5)):
  if len(layers) != len(maps):
    raise ValueError('Sizes of layers and maps do not match')
  
  plot_data1 = np.zeros((len(data1),len(layers)))
  #Load data into plotting format
  for i in range(len(data1)):
    for iter in range(len(layers)):
      plot_data1[i,iter] = data1[i][layers[iter]][maps[iter]]

  plot_data2 = np.zeros((len(data1),len(layers)))
  #Load data into plotting format
  for i in range(len(data1)):
    for iter in range(len(layers)):
      plot_data2[i,iter] = data2[i][layers[iter]][maps[iter]]
  
  if corrcoef:
    corr = np.zeros(len(layers))
    for i in range(len(layers)):
      corr[i] = stats.calc_corrcoef(data1,data2,layers[i],maps[i])

  plt.style.use('default')
  fig = plt.figure(figsize = figsize)
  ax = fig.add_subplot(111)    # The big subplot

  # Turn off axis lines and ticks of the big subplot
  ax.spines['top'].set_color('none')
  ax.spines['bottom'].set_color('none')
  ax.spines['left'].set_color('none')
  ax.spines['right'].set_color('none')
  ax.tick_params(labelcolor='w', top=False,
                 bottom=False, left=False, right=False)
  ax_2 = ax.twinx()

  ax_2.spines['top'].set_color('none')
  ax_2.spines['bottom'].set_color('none')
  ax_2.spines['left'].set_color('none')
  ax_2.spines['right'].set_color('none')
  ax_2.tick_params(labelcolor='w', top=False,
                   bottom=False, left=False, right=False)

  for p in range(len(layers)):
    ax1 = fig.add_subplot(1,len(layers),p+1)
    ax1.plot(plot_data1[:,p],'r')
    ax1.set_title('Layer: ' + str(layers[p] + 1 ) + ', Map: ' + 
                  str(maps[p] + 1) + '\n \u03C1: ' + str(corr[p]),size = 15)
    ax1.set_xticklabels(np.arange(0,7))
    ax2 = ax1.twinx()
    ax2.plot(plot_data2[:,p],'g')
  ax.set_ylabel('Tuning Value',size = 15,color = 'red')
  ax.set_xlabel('Object Category Number',size = 15)
  ax_2.set_ylabel('Gradient Value',size = 15,color = 'green')

  return fig 

def plot_corr(d1,d2,figsize = (20,8)):
  import seaborn as sns
  sns.set(style="white",rc={"lines.linewidth": 0.7})
  plt.figure(figsize = figsize)
  ax = sns.pointplot(data = d1 , color = 'mediumseagreen', errorwidth = 0.1 , capsize = 0.2)
  ax = sns.pointplot(data = d2 , color = 'orangered', errorwidth = 0.1 , capsize = 0.2)
  ax.text(0,0.4,'Regular',color = 'orangered',fontsize = 30)
  ax.text(0,0.36,'Shuffled',color = 'mediumseagreen',fontsize = 30)
  ax.set_xlabel('Layer',size = 20)
  ax.set_xticklabels(np.arange(1,14))
  ax.set_ylabel('Pearson Corr. Coeff.',size = 20)

class GraphDist() :
    def __init__(self, size, ax, x=True) :
        self.size = size
        self.ax = ax
        self.x = x

    @property
    def dist_real(self) :
        x0, y0 = self.ax.transAxes.transform((0, 0)) # lower left in pixels
        x1, y1 = self.ax.transAxes.transform((1, 1)) # upper right in pixes
        value = x1 - x0 if self.x else y1 - y0
        return value

    @property
    def dist_abs(self) :
        bounds = self.ax.get_xlim() if self.x else self.ax.get_ylim()
        return bounds[0] - bounds[1]

    @property
    def value(self) :
        return (self.size / self.dist_real) * self.dist_abs

    def __mul__(self, obj) :
        return self.value * obj
