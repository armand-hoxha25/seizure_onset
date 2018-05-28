# -*- coding: utf-8 -*-
"""
Created on Sun Jun 04 16:37:52 2017
code for feature extraction; extract required features and save the feature vector into a pickle file with the name of the directory
ictal with latency 1-15 will be denoted as pre_ictal

@author: Armand
"""

import pickle
import glob
import os
import scipy
import numpy
from scipy import signal
from sklearn import preprocessing
import matplotlib.pyplot as plt


# Take the upper right triangle of a matrix
def upper_right_triangle(matrix):
    accum = []
    for i in range(matrix.shape[0]):
        for j in range(i+1, matrix.shape[1]):
            accum.append(matrix[i, j])

    return numpy.array(accum)

def freq_correlation(fft_data):
    scaled=preprocessing.scale(fft_data,axis=0)
    corr_matrix=numpy.corrcoef(scaled)
    eigenvals=numpy.absolute(numpy.linalg.eig(corr_matrix)[0])
    eigenvals.sort()
    corr_coefficients=upper_right_triangle(corr_matrix)
    return numpy.concatenate((corr_coefficients,eigenvals))

def time_correlation(time_data):
    scaled=preprocessing.scale(time_data,axis=1)
    corr_matrix=numpy.corrcoef(scaled)
    eigenvals=numpy.absolute(numpy.linalg.eig(corr_matrix)[0])
    eigenvals.sort()
    corr_coefficients=upper_right_triangle(corr_matrix)
    return numpy.concatenate((corr_coefficients,eigenvals))

def fft_data(eeg_data,freqs):
    fft_matrix=list()
    #resampled=signal.resample(eeg_data,400,axis=1)
    resampled=eeg_data
    for ch_num in  numpy.linspace(0,resampled.shape[0]-1,resampled.shape[0]):
        fft_matrix.append(numpy.real(numpy.fft.rfft(resampled[int(ch_num)]))[freqs[0]:freqs[1]].tolist())   
    try:
        fft_matrix=numpy.log10(numpy.absolute(fft_matrix))
    except Exception:
        fft_matrix=numpy.absolute(fft_matrix)
        for ch in range(0,len(fft_matrix)-1):
            for dv in range(0,len(fft_matrix[0])-1):
                if fft_matrix[ch][dv] is 0:
                    fft_matrix[ch][dv]=1
        fft_matrix=numpy.log10(numpy.absolute(fft_matrix))
        print 1
    return fft_matrix

def process_file(file_name,freqs):
    temp_matfile=scipy.io.loadmat(file_name)
    eeg_data=temp_matfile['data']
    fft_matrix=fft_data(eeg_data,freqs)
    freq_corr=freq_correlation(fft_matrix)
    time_corr=time_correlation(eeg_data)
    features=numpy.append(freq_corr,time_corr)
    if file_name.find('test') is -1 and file_name.find('interictal') is -1:       
        if temp_matfile['latency']<16:
            intermit=file_name.find('ictal')
            new_name=file_name[0:intermit]+'early_'+file_name[intermit:]
            file_name=new_name
    else:
        file_name
    return features,file_name
    
def pickle_data(file_name,features):
    pickle_out=open(file_name+'.pickle',"wb")
    pickle.dump(features,pickle_out)
    pickle_out.close()            

#%%
#data_path='D:\documents\ICG penn state seizure\clips\ '
data_path='/media/fr3shc00kie/Mechanical_drive/documents/ICG penn state seizure/clips/'
os.chdir(data_path)
dog_path=glob.glob('Dog*')
patient_path=glob.glob('Patient*')
save_directory='/home/fr3shc00kie/Desktop/seizure onset project/pickle_data2'

freqs=[40,150] #frequencies of interest in Hz to use for seizure detection
for current_path in (dog_path+patient_path):
    os.chdir(data_path)
    os.chdir(current_path)
    all_files=glob.glob('*.mat')
    print current_path
    for file_name in all_files:
        try:
            features,save_file_name=process_file(file_name,freqs)
        except Exception:
            print file_name
        os.chdir(save_directory)
        pickle_data(save_file_name,features)
        os.chdir(data_path+current_path)
#        os.chdir(current_path)
    os.chdir('..')
    
