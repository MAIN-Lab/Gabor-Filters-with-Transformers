# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 18:34:17 2022

@author: Abel
"""

import os

from datetime import datetime 
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.layers import Concatenate, Conv2D, Conv2DTranspose, Input, MaxPool2D
from tensorflow.keras.optimizers import Adam
from keras_unet_collection2 import models, losses
from tensorflow.keras.metrics import MeanIoU
#from blocks import *
from matplotlib import pyplot as plt

from numpy import asarray
from numpy import save
from numpy import load
from supporting_functions import *
import matplotlib.pyplot as plt
import pandas as pd   
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, confusion_matrix
from model_profiler import model_profiler

import random

seed = 42
np.random.seed = seed
tf.random.set_seed(seed)

#%%
#main_path="Data"
#Dataset="MRI"
Dataset="ISLES"
#%%
#%%
#variant="_gabor_0"
#variant="_gabor_1"
#variant="_gabor_2"
#variant="_gabor_3"
variant="_gabor"
#variant=""
#%%
num_epochs=2
batchSize=1

#%%

x_train_0=load(f'{Dataset}_images_256{variant}_0_tr.npy')
x_train_1=load(f'{Dataset}_images_256{variant}_1_tr.npy')
x_train_2=load(f'{Dataset}_images_256{variant}_2_tr.npy')
x_train_3=load(f'{Dataset}_images_256{variant}_3_tr.npy')

x_test_0=load(f'{Dataset}_images_256{variant}_0_ts.npy')
x_test_1=load(f'{Dataset}_images_256{variant}_1_ts.npy')
x_test_2=load(f'{Dataset}_images_256{variant}_2_ts.npy')
x_test_3=load(f'{Dataset}_images_256{variant}_3_ts.npy')
#%%
x_train=np.concatenate([x_train_0, x_train_1, x_train_2, x_train_3], axis=-1)
x_test=np.concatenate([x_test_0, x_test_1, x_test_2, x_test_3], axis=-1)
#%%
#x_train=load(f'{Dataset}_images_256{variant}_tr.npy')
y_train=load(f'{Dataset}_Masks_256_tr.npy')
#x_test=load(f'{Dataset}_images_256{variant}_ts.npy')
y_test=load(f'{Dataset}_Masks_256_ts.npy')
#%%
if variant=="_gabor":
    x_train = x_train.astype(np.int32)
    x_test = x_test.astype(np.int32)
#y_train=y_train.astype(np.float32)
#y_test=y_test.astype(np.float32)
#%%
_, h, w, c = x_train.shape
input_shape = (h, w, c)
#%%
###sanity check
#sample=random.randint(0, len(x_train))
#plt.figure(figsize=(16, 8))
#plt.subplot(211)
#if (variant=="_gabor" or Dataset=="ISLES"):
#    plt.imshow(x_train[sample, :,:,4], 'gray') #View each channel...
#else:
#    plt.imshow(x_train[sample, :,:], 'gray') #View each channel...
#plt.subplot(212)
#plt.imshow(y_train[sample],'gray') #View each channel...
#plt.show()
#print(x_train.shape)
#print(y_train.shape)
#%% 
################ Parallel GPU#############
from tensorflow.python.client import device_lib
devices = device_lib.list_local_devices()

def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)

for d in devices:
    t = d.device_type
    name = d.physical_device_desc
    l = [item.split(':',1) for item in name.split(", ")]
    name_attr = dict([x for x in l if len(x)==2])
    dev = name_attr.get('name', 'Unnamed device')
    print(f" {d.name} || {dev} || {t} || {sizeof_fmt(d.memory_limit)}")

BATCH_SIZE = 32
#GPUS = ["GPU:0", "GPU:1", "GPU:2", "GPU:3"]
if len(devices)==2:
    GPUS = ["GPU:0"]
else:
    GPUS = ["GPU:0", "GPU:1"]
strategy = tf.distribute.MirroredStrategy( GPUS )
print('Number of devices: %d' % strategy.num_replicas_in_sync) 

batch_size = BATCH_SIZE * strategy.num_replicas_in_sync

import time

def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)

tf.get_logger().setLevel('ERROR')

################################################
#%%
evaluation_metrics=["acc", "mse", losses.dice_coef]
with strategy.scope():
    model= models.transunet_2d(input_shape, filter_num=[16, 32, 64, 128, 256], n_labels=1, stack_num_down=2, stack_num_up=2,
                                    embed_dim=768, num_mlp=3072, num_heads=4, num_transformer=4,
                                    activation='ReLU', 
                                    mlp_activation='GELU',
                                    #mlp_activation='ReLU',
                                    output_activation='Sigmoid', 
                                    batch_norm=True, pool=True, 
                                    #unpool='bilinear',
                                    unpool=True,
                                    name=('TransUnet_MHA_early'))
    model.summary()
    model.compile(optimizer=Adam(0.01), loss="binary_crossentropy", metrics=evaluation_metrics)
#%%
profile=model_profiler(model,batchSize)
print(profile)
#%%
today = datetime.now()
time_t = today.strftime("%H_%M_%S")
today_f=today.strftime('%Y%m%d')

#%%
early_stop = EarlyStopping(monitor="loss", patience=10)

checkpoint_path=f"checkpoints/model_{model.name}_{Dataset}_{today_f}_{time_t}_BestModel.hdf5"
checkpoint = ModelCheckpoint(filepath=checkpoint_path, monitor="val_dice_coef", verbose=1, save_best_only=True, mode='max')#, save_freq="epoch")
reduce_lr = ReduceLROnPlateau(monitor="loss", factor=0.3, patience=10, verbose=1, cooldown=10, min_lr=1e-5)
lr_shceduler = LearningRateScheduler(lambda _, lr: lr * np.exp(-0.01), verbose=1)
tensorboard = tf.keras.callbacks.TensorBoard("logs/1_r2a", update_freq=1)
#%%

start1 = datetime.now() 

history=model.fit(x=x_train, y=y_train, batch_size=batchSize, 
                  epochs=num_epochs, verbose=1, 
                  callbacks=[early_stop, checkpoint, reduce_lr, lr_shceduler, tensorboard], 
                  validation_split=0.10,
                  #validation_data=(x_test,y_test)
                  )
stop1 = datetime.now()
execution_time_model = stop1-start1
print(model.name+" execution time is: ", execution_time_model)
#%%
model1 = pd.DataFrame(history.history) 
historySavePath=('results/'+model.name+"_"+today.strftime('%Y%m%d')+time_t+'.csv')
with open(historySavePath, mode='w') as f:
    model1.to_csv(f)

#%%
history = model1


loss = history['loss']
val_loss = history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
loosSavePath=('results/loss_'+model.name+"_"+today.strftime('%Y%m%d')+time_t+'.png')
plt.savefig(loosSavePath)
plt.show()


acc = history['acc']
val_acc =history['val_acc']

plt.plot(epochs, acc, 'y', label='Training IoU')
plt.plot(epochs, val_acc, 'r', label='Validation IoU')
plt.title('Training and validation IoU')
plt.xlabel('Epochs')
plt.ylabel('IoU')
plt.legend()
iouSavePath=('results/acc_'+model.name+"_"+today.strftime('%Y%m%d')+time_t+'.png')
plt.savefig(iouSavePath)
plt.show()

tf.keras.backend.clear_session()


#%%
###########Testing########################################
#model = tf.keras.models.load_weights(checkpoint_path)
model.load_weights(checkpoint_path)
y_pred =(model.predict(x_test, batch_size=batchSize)[:,:,:,0] > 0.5).astype(np.uint8)
y_test=np.int_(y_test)

#%%
dice = dice_coef_f(y_test, y_pred)
jaccard= jacard_coef(y_test, y_pred)

#%%
y_test = y_test.flatten()
y_pred = y_pred.flatten()

#%%%%
start = datetime.now()

auc, f1, acc, sen, spe = image_metrics(y_test, y_pred, lim=0.5)

print(f"Accuracy = \t{acc}")
print(f"f1score\t = \t{f1}")
print(f"AUC\t = \t{auc}")
print(f"Dice\t = \t{dice}")
print(f"Jaccard\t = \t{jaccard}")
print(f"Sensitivity = \t{sen}")
print(f"Specificity = \t{spe}")

end = datetime.now()
diff = end - start
start
diff
#%%
with open(f"results/{model.name}_{Dataset}_{today_f}_{time_t}_results.txt", "w") as ff:
    ff.write(f"Accuracy = \t{acc}\n")
    ff.write(f"f1score\t = \t{f1}\n")
    ff.write(f"AUC\t = \t{auc}\n")
    ff.write(f"Dice\t = \t{dice}\n")
    ff.write(f"Jaccard\t = \t{jaccard}\n")
    ff.write(f"Sensitivity = \t{sen}\n")
    ff.write(f"Specificity = \t{spe}\n")
#%%
test_pred_batch = (model.predict(x_test, batch_size=batchSize)[:,:,:,0] > 0.5).astype(np.uint8)
#%%
n_classes = 2
IOU_keras = MeanIoU(num_classes=n_classes)  
IOU_keras.update_state(test_pred_batch, y_test)
print("Mean IoU =", IOU_keras.result().numpy())

