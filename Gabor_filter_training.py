# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 14:24:45 2022

@author: Abel
"""

import numpy as np
import cv2
import pandas as pd
from matplotlib import pyplot as plt
from numpy import save
from numpy import load
import random
 #%%
#img = cv2.imread('BSE_Image.jpg')
#%%
dataset="MRI"
image_numpyFileName_train=f"{dataset}_images_64_tr.npy"
image_numpyFileName_train_Gabor=f"{dataset}_images_64_tr_gabor.npy"
#%%
#image_numpyFileName_train="x_train_CHASEDB1.npy"
#image_numpyFileName_train_Gabor="x_train_CHASEDB1_gabor.npy"


#%%
x_train=load(image_numpyFileName_train)

#x_train=x_train[0:3,:,:,:]

#y_train=load(mask_numpyFileName_train)
#%%
#img =x_train[0,:,:,:]
#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
#Here, if you have multichannel image then extract the right channel instead of converting the image to grey. 
#For example, if DAPI contains nuclei information, extract the DAPI channel image first. 

#Multiple images can be used for training. For that, you need to concatenate the data
#%%
#Save original image pixels into a data frame. This is our Feature #1.



#Generate Gabor features
#num = 1  #To count numbers up in order to give Gabor features a lable in the data frame
  #Create empty list to hold all kernels that we will generate in a loop
#filter_image=[]
#filter_image=np.empty(shape=(576,576,1),dtype='uint8')
#plt.imshow(filter_image)

#%%
def gabor_filterBank(img):
    #df = pd.DataFrame()
    num=1
    kernels = []
    for theta in range(2):   #Define number of thetas. Here only 2 theta values 0 and 1/4 . pi 
        theta = theta / 4. * np.pi
        for sigma in (1, 3):  #Sigma with values of 1 and 3
            for lamda in np.arange(np.pi / 4, 2*np.pi, np.pi / 4):   #Range of wavelengths
                for gamma in (0.05, 0.5):   #Gamma values of 0.05 and 0.5
                    
                    gabor_label = 'Gabor' + str(num)  #Label Gabor columns as Gabor1, Gabor2, etc.
    #                print(gabor_label)
                    ksize=9
                    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)    
                    kernels.append(kernel)
                    #Now filter the image and add values to a new column 
                    fimg = cv2.filter2D(img, cv2.CV_8UC3, kernel)
                    fimg_ext = np.expand_dims((np.array(fimg)),2)
                    fimg_ext=fimg_ext.tolist()
                    if num==1:
                        filter_image=fimg_ext
                    else:
                        filter_image=np.append(filter_image,fimg_ext,axis=2)
                    #filter_image.append(fimg_ext)
                    #filter_image=np.append(filter_image,fimg_ext,axis=2)
                    #filtered_img = fimg.reshape(-1)
#                    cv2.imshow('Filtered', fimg)
#                    cv2.waitKey(100)
#                    cv2.destroyAllWindows()
                    #df[gabor_label] = filtered_img  #Labels columns as Gabor1, Gabor2, etc.
                    print(gabor_label, ': theta=', theta, ': sigma=', sigma, ': lamda=', lamda, ': gamma=', gamma)
                    num += 1  #Increment for gabor column label
    return filter_image
#%%
filtered_images_training=[]
count=1
for i in range(x_train.shape[0]):
    img=x_train[i,:,:,:]
    img_gabor=gabor_filterBank(img)
    filtered_images_training.append(img_gabor)
    print(f"Image # {count}")
    count+=1
#%%
filtered_images_training = np.asarray(filtered_images_training) 
np.save(image_numpyFileName_train_Gabor,filtered_images_training)
#%%
test_gaborBank=np.load(image_numpyFileName_train_Gabor)
#%%
sample=random.randint(0, len(test_gaborBank))
img_test=test_gaborBank[sample]

fig = plt.figure(figsize=(15, 12))
for idx in range(len(img_gabor)):
    ax = fig.add_subplot(len(img_gabor)/8, 8, idx+1) # this line adds sub-axes
    ax.axis('off')
    ax.imshow(img_test[:,:,idx], 'gray') # this line creates the image using the pre-defined sub axes

#%%
#x_train_sample=x_train[sample]
#plt.imshow(x_train_sample, 'gray')
#
##%%
#y_train=np.load("y_train_CHASEDB1.npy")
##%%
#y_train_sample=y_train[sample]
#plt.imshow(y_train_sample, 'gray')

#%%






##%%
#    for theta in range(2):   #Define number of thetas. Here only 2 theta values 0 and 1/4 . pi 
#        theta = theta / 4. * np.pi
#        for sigma in (1, 3):  #Sigma with values of 1 and 3
#            for lamda in np.arange(0, np.pi, np.pi / 4):   #Range of wavelengths
#                for gamma in (0.05, 0.5):   #Gamma values of 0.05 and 0.5
#                    
#                    gabor_label = 'Gabor' + str(num)  #Label Gabor columns as Gabor1, Gabor2, etc.
#    #                print(gabor_label)
#                    ksize=9
#                    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)    
#                    kernels.append(kernel)
#                    #Now filter the image and add values to a new column 
#                    fimg = cv2.filter2D(img, cv2.CV_8UC3, kernel)
#                    fimg_ext = np.expand_dims((np.array(fimg)),2)
#                    fimg_ext=fimg_ext.tolist()
#                    if num==1:
#                        filter_image=fimg_ext
#                    else:
#                        filter_image=np.append(filter_image,fimg_ext,axis=2)
#                    #filter_image.append(fimg_ext)
#                    #filter_image=np.append(filter_image,fimg_ext,axis=2)
#                    #filtered_img = fimg.reshape(-1)
#                    cv2.imshow('Filtered', fimg)
#                    cv2.waitKey(100)
#                    cv2.destroyAllWindows()
#                    #df[gabor_label] = filtered_img  #Labels columns as Gabor1, Gabor2, etc.
#                    print(gabor_label, ': theta=', theta, ': sigma=', sigma, ': lamda=', lamda, ': gamma=', gamma)
#                    num += 1  #Increment for gabor column label
#
##filter_image_arr = np.asarray(filter_image)        
#        
#print(df.head())
#
#df.to_csv("Gabor.csv")
#
##%%
#
#fig = plt.figure(figsize=(15, 12))
#
#
#
#for idx in range(32):
#    ax = fig.add_subplot(4, 8, idx+1) # this line adds sub-axes
#    ax.axis('off')
#    ax.imshow(filter_image[:,:,idx], 'gray') # this line creates the image using the pre-defined sub axes
##fig.subplots_adjust(hspace=0, wspace=0.2)
##fig.tight_layout()
#
#
#
#
#
#
#






