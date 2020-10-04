#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 01:27:25 2020

@author: soukhind
"""
import tensorflow as tf
from keras.models import Model
from keras.layers import Flatten, Dense, Dropout
from keras.layers import Input
import numpy as np

def make_gradcam_heatmap(
    img_array, model, layer_name, classifier_layer_names,top_model):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer
    top_layer_names = [layer.name for layer in top_model.layers]
    conv_layer = model.get_layer(layer_name)
    conv_layer_model = Model(model.inputs, conv_layer.output)

    # Second, we create a model that maps the activations of the last conv
    # layer to the final class predictions
    classifier_input = Input(shape=conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in classifier_layer_names:
        if layer_name in top_layer_names:
          x = top_model.get_layer(layer_name)(x)
        else:
          x = model.get_layer(layer_name)(x)
    classifier_model = Model(classifier_input, x)

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        # Compute activations of the last conv layer and make the tape watch it
        conv_layer_output = conv_layer_model(img_array)
        tape.watch(conv_layer_output)
        # Compute class predictions
        preds = classifier_model(conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]

    # This is the gradient of the top predicted class with regard to
    # the output feature map of the last conv layer
    grads = tape.gradient(top_class_channel, conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    conv_layer_output = conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        conv_layer_output[:, :, i] *= pooled_grads[i]

    # The channel-wise mean of the resulting feature map
    # is our heatmap of class activation
    heatmap = np.mean(conv_layer_output, axis=-1)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    #heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return pooled_grads,heatmap

def reduce_catgrad_mean(data):
  reduce_data = [[0 for item in subl] for subl in data[0]]
  for img in range(len(data)):
    for layer in range(len(data[0])):
      for map in range(len(data[0][layer])):
        reduce_data[layer][map] += data[img][layer][map]/len(data)
  return reduce_data

def calc_avg_gradcam(data,model,top_model):
  grad_cat = [[] for i in range(len(data))] #Initialize gradients for each image in cat
  for img in range(len(data)): # Parse each image
    for idx,layer_name in enumerate(layer_names_all):
      if 'conv' in layer_name:
        classifier_layer_names = layer_names_all[idx + 1:] + top_layer_names
        img_tensor = data[img].reshape([1,224,224,3])
        plt.imshow(data[img])
        g,h = make_gradcam_heatmap(img_tensor,model,layer_name,classifier_layer_names,top_model)
        grad_cat[img].append(g) #grads has dimensions of layers x no of maps in each layer
    print(img)
  grads = reduce_catgrad_mean(grad_cat) # Reduce to the mean of all the images in cat
  del grad_cat
  return grads