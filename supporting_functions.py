# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 18:44:12 2022

@author: Abel
"""
#%%
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras import backend as k


#%%

def dice_coef(y_true, y_pred):
    smooth = 0
    y_true_f = k.flatten(y_true)
    y_pred_f = k.flatten(y_pred)
    intersection = k.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (k.sum(y_true_f) + k.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def dice_coef_f(y_true, y_pred):
    smooth = 0
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def jacard_coef(y_true, y_pred):
    y_true_f = y_true.flatten()
    #y_true_f=y_true_f.astype('float32')
    y_pred_f = y_pred.flatten()
    #y_pred_f=y_pred_f.astype('float32')
    intersection = np.sum(y_true_f * y_pred_f)
    
    return (intersection + 1.0) / (np.sum(y_true_f) + np.sum(y_pred_f) - intersection + 1.0)


def jacard_coef_loss(y_true, y_pred):
    return -jacard_coef(y_true, y_pred)  # -1 ultiplied as we want to minimize this value as loss function
#%%
def image_metrics(Y_test, Y_pred, lim=0.5):
    Y_pred_bin = np.zeros_like(Y_test)
    idx = Y_pred > lim
    Y_pred_bin[idx] = 1

    auc = roc_auc_score(Y_test, Y_pred)
    f1 = f1_score(Y_test, Y_pred_bin)
    acc = accuracy_score(Y_test, Y_pred_bin)
    mat = confusion_matrix(Y_test, Y_pred_bin)

    TN = mat[0][0]
    FN = mat[1][0]
    TP = mat[1][1]
    FP = mat[0][1]

    sen = TP / (TP + FN)
    spe = TN / (TN + FP)

    return auc, f1, acc, sen, spe