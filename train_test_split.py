# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 20:20:24 2017

@author: Armand

train test split data

"""
import glob
import os
#from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
import pickle
#%%
def unpickle_feature(datafile):   
    pickle_in=open(datafile,"rb")
    features=pickle.load(pickle_in)
    return features
#%% 
def subject_features(subject,feature_folder):                   
    feature_folder=feature_folder+"//".strip()
    early_features_files=glob.glob(feature_folder+subject+'_early*')
    ictal_features_files=glob.glob(feature_folder+subject+'_ictal*')
    interictal_features_files=glob.glob(feature_folder+subject+'_interictal*')
    early_features=list()
    ictal_features=list()
    interictal_features=list()
    
    for datafile in early_features_files:
        early_features.append(unpickle_feature(datafile))
    for datafile in ictal_features_files:
        ictal_features.append(unpickle_feature(datafile))
    for datafile in interictal_features_files:
        interictal_features.append(unpickle_feature(datafile))
    print("number of featuers ictal",len(ictal_features_files))
    print("number of featuers interictal",len(interictal_features_files))
    print("number of featuers early",len(early_features_files))
    return early_features, ictal_features, interictal_features
#%%                            
def train_test_split_data(feature_folder,subject,perc_train):
    #return x_train,y_train,x_test,y_test
    #perc_train from 0 to 1 for percentage of data to be used in training rather than testing
    early_features,ictal_features,interictal_features=subject_features(subject,feature_folder)
    n_features=len(early_features)+len(ictal_features)+len(interictal_features)
    y=list()
    x=early_features+ictal_features+interictal_features   
    for rand_a in early_features:
        y.append(0)
    for rand_a in ictal_features:
        y.append(1)
    for rand_a in interictal_features:
        y.append(2)
    x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=perc_train,random_state=1)
    return x_train,x_test,y_train,y_test

#x_train, x_test, y_train, y_test=train_test_split_data('C:\Users\Armand\Google Drive\seizure detection project\pickle data','Dog_1',0.5)
        