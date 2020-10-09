#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 00:52:15 2020

@author: soukhind
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
    #img1 /= 255.
    #img2 /= 255.
    ovy = (img1 + img2)/2
    return ovy



def generate_merged(src,destn,nmergs):
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
    os.makedirs(destn + '/Female',exist_ok=True)
    os.makedirs(destn + '/Manmade',exist_ok=True)
    os.makedirs(destn + '/Natural',exist_ok=True)
    os.makedirs(destn + '/Powered',exist_ok=True)
    os.makedirs(destn + '/Nonpowered',exist_ok=True)
    os.makedirs(destn + '/Male',exist_ok=True)
    
    dirs = []
    for dirname,_,_ in os.walk(src):
      dirs.append(dirname)
    dirs = [dirs[i] for i in [2,3,5,6,8,9]] # extract the required sublcass dirs
    savestrs = ['Natural','Manmade','Nonpowered','Powered','Male','Female']
    #savestrs is positioned according to dirs
    
    cats = np.arange(0,6)
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

def sort_imgs(data_path,train_path,test_path,correct = True):
    """
    

    Parameters
    ----------
    data_path : str
        Source path to load images.
    train_path : str
        Path to store training images.
    test_path : str
        Path to store test images.

    Returns
    -------
    None.

    """
    
    # 150 images in total. 
    # 120 Train, 30 Test
    # 75 Correct,75 Incorrect for Train:15 incorrect from other cats
    # 15 Correct,15 Incorrect for Test:3 incorrect from each cats
    if correct == True:
        train_path =  train_path + '/Correct'
        test_path = test_path + '/Correct'
    else:
        train_path =  train_path + '/Incorrect'
        test_path = test_path + '/Incorrect'
    
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
    
    if correct:
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
    else:
        
        # Walk through and index all the files
        dirname = []
        files = [[] for i in range(6)]
        i = -1
        for dpath,_,f in os.walk(data_path):
            if len(files) > 1:
                dirname.append(dpath)
                files[i] = f
                i += 1
                
        dirname = dirname[1:]
        
        for item in files:
            for vals  in item:
                if vals == '.DS_Store':
                    item.remove('.DS_Store')
        
        #Now start saving incorrect ones
        cats = np.arange(0,6)
        for cat in cats:
            wdir1 = savepaths1[cat]
            wdir2 = savepaths2[cat]
            mdir = dirname[cat]
            temp = np.array(dirname)
            odirs = temp[temp!=mdir]
            for idx,dirs in enumerate(odirs):
                fset = np.random.permutation(files[dirname.index(dirs)])
                for tr in fset[0:15]:
                        shutil.copyfile(os.path.join(dirs,tr),os.path.join(wdir1,tr))
                for tt in fset[15:18]:
                        shutil.copyfile(os.path.join(dirs,tt),os.path.join(wdir2,tt))
            

            
#%%
train_path = '/Users/soukhind/Desktop/ann/data/merge/merge_train/Incorrect'
test_path = '/Users/soukhind/Desktop/ann/data/merge/merge_test/Incorrect'
savepaths1 = []
for dirname,_,_ in os.walk(train_path):
    savepaths1.append(dirname)
savepaths1 = savepaths1[1:]

savepaths2 = []
for dirname,_,_ in os.walk(test_path):
    savepaths2.append(dirname)
savepaths2 = savepaths2[1:]

# Walk through and index all the files
dirname = []
files = [[] for i in range(6)]
i = -1
for dpath,_,f in os.walk('/Users/soukhind/Desktop/ann/data/merge/merge_data'):
    if len(files) > 1:
        dirname.append(dpath)
        files[i] = f
        i += 1
        
dirname = dirname[1:]

for item in files:
    for vals  in item:
        if vals == '.DS_Store':
            item.remove('.DS_Store')

#Now start saving incorrect ones
cats = np.arange(0,6)
for cat in cats:
    wdir1 = savepaths1[cat]
    wdir2 = savepaths2[cat]
    mdir = dirname[cat]
    temp = np.array(dirname)
    odirs = temp[temp!=mdir]
    for idx,dirs in enumerate(odirs):
        print(idx)
        fset = np.random.permutation(files[dirname.index(dirs)])
        for tr in fset[0:15]:
                shutil.copyfile(os.path.join(dirs,tr),os.path.join(wdir1,tr))
        for tt in fset[15:18]:
                shutil.copyfile(os.path.join(dirs,tt),os.path.join(wdir2,tt))
    

   


    
        
