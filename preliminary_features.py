import glob
import os
import pandas
import scipy.io
import seaborn
import matplotlib
import numpy
import data_exploration as de
import platform
import sklearn.preprocessing

#%%
if platform.system() is 'Windows':
    DATA_DIR='E:\ICG penn state seizure\clips'
else:
    DATA_DIR='/media/fr3shc00kie/Mechanical_drive/documents/ICG penn state seizure/clips'

SUB_LIST=['Dog_1','Dog_2','Dog_3','Dog_4', 'Patient_1',
          'Patient_2', 'Patient_3', 'Patient_4', 'Patient_5', 'Patient_6', 'Patient_7', 'Patient_8']

#get list of files for each subject with full path

sub_list=list()
sub_files=list()
file_type=list()
srate=list()
nchan=list()

alpha_power=list()
beta_power=list()
gamma_power=list()

std_time=list()
median_time=list()
max_time=list()
min_time=list()
time_corr=list()
freq_corr=list()
time_eig=list()
freq_eig=list()

for subject in SUB_LIST:

#    if platform.system() is 'Windows':
    all_sub_files=glob.glob(os.path.join(DATA_DIR,subject)+"\*.mat")
#    else:
#        DATA_DIR='/media/fr3shc00kie/Mechanical_drive/documents/ICG penn state seizure/clips/'
  
    print 'Processing: ',subject

    for current_file in all_sub_files:
        temp_mat=scipy.io.loadmat(current_file)
        temp_mat['data']=sklearn.preprocessing.scale(temp_mat['data'],axis=0)
        srate.append(temp_mat['freq'])
        in0=numpy.ndarray.tolist(temp_mat['channels'])
        in1=in0[0]
        nchan.append(len(in1[0]))
        sub_list.append(subject)
        sub_files.append(os.path.abspath(current_file))

        #obtain alpha_power,beta_power,gamma_power area under curve
        fft,freqs=de.fft_data(temp_mat)
        alpha_power.append(de.alpha_pow(fft,freqs))
        beta_power.append(de.beta_pow(fft,freqs))
        gamma_power.append(de.gamma_pow(fft,freqs))

        #get std_time, medina_time,max_time,min_time,range_time
        time_data=(temp_mat['data'])
        time_1d=time_data.reshape(1,time_data.shape[0]*time_data.shape[1])
        std_time.append(numpy.std(time_1d))
        median_time.append(numpy.median(time_1d))
        max_time.append(numpy.max(time_1d))
        min_time.append(numpy.min(time_1d))

        #mean correlation in time
        try:
            time_corr.append(numpy.mean(de.time_correlation(temp_mat['data'])[0]))
            time_eig.append(numpy.mean(de.time_correlation(temp_mat['data'])[1]))
        except Exception:
            time_corr.append(0)
            time_eig.append(0)
        #mean correlation in frequency
        freq_corr.append(numpy.mean(de.freq_correlation(fft)[0]))
        freq_eig.append(numpy.mean(de.freq_correlation(fft)[1]))
        if 'interictal' in current_file:
            file_type.append('interictal')
        elif 'ictal' in current_file:
            if temp_mat['latency']<16:
                file_type.append('early')
            else:
                file_type.append('ictal')
        else:
            file_type.append('test')
#%% making dataframe
import pickle
           
big_dict=dict()
alpha_power2=numpy.asarray(alpha_power)
big_dict['subject']=sub_list
big_dict['file']=sub_files
big_dict['alpha']=numpy.mean(alpha_power)
big_dict['beta']=numpy.mean(beta_power)
big_dict['gamma']=numpy.mean(gamma_power)
big_dict['std_time']=std_time
big_dict['median_time']=median_time
big_dict['max_time']=max_time
big_dict['min_time']=min_time
big_dict['time_corr']=time_corr
big_dict['time_eig']=time_eig
big_dict['freq_corr']=freq_corr
big_dict['freq_eig']=freq_eig
big_dict['type']=file_type
big_dict['srate']=srate
big_dict['nchan']=nchan

DF=pandas.DataFrame(big_dict)

with open('subject_dataframe1.pickle','wb') as handle:
    pickle.dump(DF,handle,protocol=pickle.HIGHEST_PROTOCOL)

