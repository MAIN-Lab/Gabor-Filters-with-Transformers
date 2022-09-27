# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 23:04:01 2022

@author: Abel
"""

import tensorflow as tf
from tensorflow import keras
from keras_unet_collection import models
import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from datetime import datetime 
import cv2
from PIL import Image
from numpy import load
from keras import backend, optimizers
from sklearn.metrics import f1_score, jaccard_score
from sklearn.model_selection import train_test_split
from focal_loss import BinaryFocalLoss, sparse_categorical_focal_loss
import random
from tensorflow.keras.metrics import MeanIoU
from sklearn.preprocessing import MinMaxScaler

from models import Attention_ResUNet, UNet, Attention_UNet, dice_coef, dice_coef_loss, jacard_coef, Attention_UNet_Plus
#%%
scaler = MinMaxScaler()


#%%

images_gabor=load("MRI_images_64_tr_gabor.npy")
images=load("MRI_images_64_tr.npy")
#%%
#images_gabor=scaler.fit_transform(images_gabor.reshape(-1, images_gabor.shape[-1])).reshape(images_gabor.shape)


#%%
images=images*255.
#%%
images_gabor=images_gabor.astype(np.float32)
images=images.astype(np.float32)
#%%
images_tr=[]
count=1
for img in range(images.shape[0]):
    image_training=np.append(images_gabor[img],images[img],axis=2)
    images_tr.append(image_training)
    count+=1
    print(f"Image concatenate {count}")
    
#%%    
images_trainingFileName="MRI_images_64_gabor_tr.npy"
images_tr_arr = np.asarray(images_tr,dtype='float32')
#%% 
np.save(images_trainingFileName,images_tr_arr)



################################

#%%
#%%

images_gabor=load("MRI_images_64_ts_gabor.npy")
images=load("MRI_images_64_ts.npy")
#%%
#images_gabor=scaler.fit_transform(images_gabor.reshape(-1, images_gabor.shape[-1])).reshape(images_gabor.shape)

#%%
images=images*255.
#%%
images_gabor=images_gabor.astype(np.float32)
images=images.astype(np.float32)

#%%
images_ts=[]
count=1
for img in range(images.shape[0]):
    image_testing=np.append(images_gabor[img],images[img],axis=2)
    images_ts.append(image_testing)
    count+=1
    print(f"Image concatenate {count}")
    
#%%    
images_testingFileName="MRI_images_64_gabor_ts.npy"
images_ts_arr = np.asarray(images_ts, dtype='float32')
#%% 
np.save(images_testingFileName,images_ts_arr)