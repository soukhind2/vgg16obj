#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 00:52:15 2020

@author: soukhind

This file calculates the overlaid images, sorts them and saves them
in respective folders.
Strict requirment: Folder structure
"""
import cv2
import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img,img_to_array
import shutil

path = '/Users/soukhind/Desktop/ann/data/train'
nmergs = 20 #number of merged images to generate for each cat
savepath = 'merge_data/'


def overlay(path1,path2):
    """
    function to overlay the images for the 'merged' category

    Parameters
    ----------
    path1 : str
        path to first image.
    path2 : str
        path to second image.
    Order of the paths do not matter
    Returns
    -------
    The average of two images.

    """
    
    img1 = img_to_array(load_img(path1,target_size = (224,224)))
    img2 = img_to_array(load_img(path2,target_size = (224,224)))
    #img1 /= 255. # Opencv does not like floating point values
    #img2 /= 255.
    ovy = (img1 + img2)/2
    return ovy



def generate_merged(src,destn,nmergs,correct = True):
    """
    

    Parameters
    ----------
    src : str
        source path.
        the path should be presorted to include the six subclass folders
    destn : str
        destination path.
    nmergs : int
        number of merged images to generate per category.

    Returns
    -------
    None.

    """
    os.makedirs(destn,exist_ok=True)
    if correct == True:        
        os.makedirs(destn + '/Correct' + '/Female',exist_ok=True)
        os.makedirs(destn + '/Correct' + '/Manmade',exist_ok=True)
        os.makedirs(destn + '/Correct' + '/Natural',exist_ok=True)
        os.makedirs(destn + '/Correct' + '/Powered',exist_ok=True)
        os.makedirs(destn + '/Correct' + '/Nonpowered',exist_ok=True)
        os.makedirs(destn + '/Correct' + '/Male',exist_ok=True)
        destn = destn + '/Correct'
    else:
        os.makedirs(destn + '/Incorrect' + '/Female',exist_ok=True)
        os.makedirs(destn + '/Incorrect' + '/Manmade',exist_ok=True)
        os.makedirs(destn + '/Incorrect' + '/Natural',exist_ok=True)
        os.makedirs(destn + '/Incorrect' + '/Powered',exist_ok=True)
        os.makedirs(destn + '/Incorrect' + '/Nonpowered',exist_ok=True)
        os.makedirs(destn + '/Incorrect' + '/Male',exist_ok=True)        
        destn = destn + '/Incorrect/'

    dirs = []
    for dirname,_,_ in os.walk(src):
      dirs.append(dirname)
    dirs = [dirs[i] for i in [2,3,5,6,8,9]] # extract the required sublcass dirs
    savestrs = ['Natural','Manmade','Nonpowered','Powered','Male','Female']
    #savestrs is positioned according to dirs

    cats = np.arange(0,6)
    if correct:
        for cat in cats:
          for img in range(nmergs):
            # pick out the other cat to merge
            other_cat = np.random.choice(cats[cats != cat]) 
            while True:
              # select a random file from first cat
              fname1 = np.random.choice(os.listdir(dirs[cat])) 
              path1 = os.path.join(dirs[cat], fname1)
              # select a random file from second cat
              fname2 = np.random.choice(os.listdir(dirs[other_cat]))
              path2 = os.path.join(dirs[other_cat], fname2)
              if fname1 != '.DS_Store' and  fname2 != '.DS_Store':  # Exclude unwanted files 
                break
            oly = overlay(path1,path2) # generate the overlay
            filepath = str(os.path.join(destn,savestrs[cat]) + '/' +
                           savestrs[cat] + '_' + str(img+1) + '.jpg')
            cv2.imwrite(filepath,
                        cv2.cvtColor(oly, cv2.COLOR_RGB2BGR)) # flip the rgb channels
    
    else:
        srcdir = []
        for dpath,_,f in os.walk(src):
            if len(f) > 1:
                srcdir.append(dpath)
    
        for cat in range(6):
            temp1 = srcdir.pop(cat)
            tail_path = os.path.split(temp1)[1]
            wdir = os.path.join(destn,tail_path)
            for img in range(nmergs):

                cat1dir,cat2dir = np.random.choice(srcdir,2,replace = False)
                while True:
                    fname1 = np.random.choice(os.listdir(cat1dir)) 
                    path1 = os.path.join(cat1dir, fname1)
                    # select a random file from second cat
                    fname2 = np.random.choice(os.listdir(cat2dir))
                    path2 = os.path.join(cat2dir, fname2)
                    if fname1 != '.DS_Store' and  fname2 != '.DS_Store':  
                        # Exclude unwanted files 
                        break
                #print(path1,path2)
                oly = overlay(path1,path2) # generate the overlay
                filepath = str(wdir + '/' +
                           savestrs[cat] + '_inc' + str(img+1) + '.jpg')
                #print(filepath)
                cv2.imwrite(filepath,
                        cv2.cvtColor(oly, cv2.COLOR_RGB2BGR)) # flip the rgb channels
            srcdir.insert(cat, temp1)
            print(cat)
            
            
def sort_imgs(data_path,train_path,test_path,correct,n_train = 75,n_test = 15):
    """
    

    Parameters
    ----------
    data_path : str
        Source path to load images.
        The source must contain correct and incorrect subfolders
        Each subfolders must contain the class merged images
    train_path : str
        Path to store training images.
    test_path : str
        Path to store test images.
    n_train : int
        Number of trianing images to sort
    n_test : int
        Number of testing images to sort
    Returns
    -------
    None.

    """
    
    # 150 images in total. 
    # 120 Train, 30 Test
    # 75 Correct,75 Incorrect for Train:15 incorrect from other cats
    # 15 Correct,15 Incorrect for Test:3 incorrect from each cats
    
    if correct == True:
        train_path =  train_path + '/Correct/'
        test_path = test_path + '/Correct/'
        data_path = data_path + '/Correct/'
    else:
        train_path =  train_path + '/Incorrect/'
        test_path = test_path + '/Incorrect/'
        data_path = data_path + '/Incorrect/'
    
    os.makedirs(train_path,exist_ok=True)
    os.makedirs(str(train_path + '/Female'),exist_ok=True)
    os.makedirs(str(train_path + '/Manmade'),exist_ok=True)
    os.makedirs(str(train_path + '/Natural'),exist_ok=True)
    os.makedirs(str(train_path + '/Powered'),exist_ok=True)
    os.makedirs(str(train_path + '/Nonpowered'),exist_ok=True)
    os.makedirs(str(train_path + '/Male'),exist_ok=True)
    
    os.makedirs(test_path,exist_ok=True)
    os.makedirs(test_path + '/Female',exist_ok=True)
    os.makedirs(test_path + '/Manmade',exist_ok=True)
    os.makedirs(test_path + '/Natural',exist_ok=True)
    os.makedirs(test_path + '/Powered',exist_ok=True)
    os.makedirs(test_path + '/Nonpowered',exist_ok=True)
    os.makedirs(test_path + '/Male',exist_ok=True)
    
    savepaths1 = []
    for dirname,_,_ in os.walk(train_path):
        savepaths1.append(dirname)
    savepaths1 = savepaths1[1:]
    
    savepaths2 = []
    for dirname,_,_ in os.walk(test_path):
        savepaths2.append(dirname)
    savepaths2 = savepaths2[1:]
    

    i = -1
    for dirpath,dirname,files in os.walk(data_path):
        n = len(files)
        if n > 1:
            file_set = files
            np.random.permutation(file_set)
            train = file_set[0:75] #Selecting correct images
            test = file_set[75:90] #Selecting correct images
            for tr in train:
                shutil.copyfile(os.path.join(dirpath,tr),os.path.join(savepaths1[i],tr))
            for tt in test:
                shutil.copyfile(os.path.join(dirpath,tt),os.path.join(savepaths2[i],tt))
        i += 1

            