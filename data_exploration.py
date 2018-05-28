from __future__ import division
import numpy
from sklearn import preprocessing
import matplotlib as plt

def fft_data(temp_mat):
    fft_matrix=list()
    #resampled=signal.resample(eeg_data,400,axis=1)
    resampled=temp_mat['data']
    for ch_num in  numpy.linspace(0,resampled.shape[0]-1,resampled.shape[0]):
        fft_matrix.append(numpy.real(numpy.fft.rfft(resampled[int(ch_num)])).tolist())   
        

    fft_matrix=numpy.absolute(fft_matrix)
    for ch in range(0,len(fft_matrix)-1):
        for dv in range(0,len(fft_matrix[0])-1):
            if fft_matrix[ch][dv] in [numpy.inf,-numpy.inf]:
                print fft_matrix[ch][dv],'found at', ch,dv
                fft_matrix[ch][dv]=1
    #fft_matrix=numpy.log10(numpy.absolute(fft_matrix))
    
    freq=numpy.fft.fftfreq(resampled.shape[1])*numpy.round(temp_mat['freq'])
    fft_matrix2=list()
    for ch in range(0,len(fft_matrix)-1):
        fft_matrix2.append(fft_matrix[ch][1:49])
    fft_matrix2=numpy.real(numpy.asarray(fft_matrix2))
    return fft_matrix2,numpy.round(freq[1:49])

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
    return corr_coefficients,eigenvals

def time_correlation(time_data):
    scaled=preprocessing.scale(time_data,axis=1)
    corr_matrix=numpy.corrcoef(scaled)
    eigenvals=numpy.absolute(numpy.linalg.eig(corr_matrix)[0])
    eigenvals.sort()
    corr_coefficients=upper_right_triangle(corr_matrix)
    return corr_coefficients,eigenvals

'''
def fft_freq(temp_mat):
    data=temp_mat['data']
    data_1d=data.reshape(1,data.shape[0]*data.shape[1])
    fft=numpy.fft.fft(data_1d)
    freq=numpy.fft.fftfreq(data.shape[1])*numpy.round(temp_mat['freq'])
    return fft[0], numpy.round(freq)
'''
def alpha_pow(fft,freq):
    f1=8
    f2=12
    found=0
    j1=0
    while found==0:
        if freq[j1]==f1:
            found=1
            f1_ind=j1
        j1=j1+1

    found=0
    j1=0
    while found==0:
        if freq[j1]==f2:
            found=1
            f2_ind=j1
        j1=j1+1

    alpha_pow=numpy.mean(numpy.trapz(numpy.real(fft)[:,f1_ind:f2_ind]))
    return alpha_pow


def beta_pow(fft,freq):
    f1=12
    f2=30
    found=0
    j1=0
    while found==0:
        if freq[j1]==f1:
            found=1
            f1_ind=j1
        j1=j1+1

    found=0
    j1=0
    while found==0:
        if freq[j1]==f2:
            found=1
            f2_ind=j1
        j1=j1+1

    beta_pow=numpy.mean(numpy.trapz(numpy.real(fft)[:,f1_ind:f2_ind]))
    return beta_pow


def gamma_pow(fft,freq):
    f1=30
    f2=47
    found=0
    j1=0
    while found==0:
        if freq[j1]==f1:
            found=1
            f1_ind=j1
        j1=j1+1

    found=0
    j1=0
    while found==0:
        if freq[j1]==f2:
            found=1
            f2_ind=j1
        j1=j1+1

    gamma_pow=numpy.mean(numpy.trapz(numpy.real(fft)[:,f1_ind:f2_ind]))
    return gamma_pow

#%% sampling rates of each subject
def get_srates(DF):        
    srate_list=list()
    SUB_LIST=list(set(DF.subject))
    for subject in SUB_LIST:
        #pull df of only current subject, and make unique list of srates
        #so each subject has a list of unique sampling rates
        subject_df=DF[DF.subject==subject]
        subject_df.srate.apply(set)
        temp_list=list()
        for index in subject_df.index:
            temp_list.append([subject_df.loc[index].subject,
                              subject_df.loc[index].srate[0],subject_df.loc[index].nchan])
        set_list=set(tuple(i) for i in temp_list)
        srate_list.append(list(set_list))
        
    print srate_list
    return srate_list

#%% loading saved dataframe
def load_DF():
    import pickle
    with open('subject_dataframe1.pickle','rb') as handle:
        DF=pickle.load(handle)
        
    return DF

#%% stack bar plot
def plot_stack_type(DF):
    # libraries
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import rc
    import pandas as pd
    import matplotlib
    plt.figure(figsize=(8,13))
    plt.subplot(111, facecolor='w')
    font = {'family' : 'normal',
            'size'   : 15}
    
    matplotlib.rc('font', **font)
    # Data
    #raw_data = {'greenBars': [20, 1.5, 7, 10, 5], 'orangeBars': [5, 15, 5, 10, 15],'blueBars': [2, 15, 18, 5, 10]}
    #df = pd.DataFrame(raw_data)
    
    sub_list=list(set(DF.subject))
    early_list=list()
    ictal_list=list()
    interictal_list=list()
    
    for subject in sub_list:
        early_list.append(len(DF.loc[(DF.subject==subject) & (DF.type=='early')]))
        ictal_list.append(len(DF.loc[(DF.subject==subject) & (DF.type=='ictal')]))
        interictal_list.append(len(DF.loc[(DF.subject==subject) & (DF.type=='interictal')]))
    
    raw_data={'early': early_list, 'ictal':ictal_list, 'interictal':interictal_list}
    df=pd.DataFrame(raw_data)
    r = range(0,len(sub_list))
     
    # From raw value to percentage
    totals = [i+j+k for i,j,k in zip(df['early'], df['ictal'], df['interictal'])]
    greenBars = [i / j * 100 for i,j in zip(df['early'], totals)]
    orangeBars = [i / j * 100 for i,j in zip(df['ictal'], totals)]
    blueBars = [i / j * 100 for i,j in zip(df['interictal'], totals)]
     
    # plot
    barWidth = 0.85
    names = sub_list
    
    # Create green Bars
    plt.bar(r, greenBars, color='#b5ffb9', edgecolor='white', width=barWidth, label="early")
    # Create orange Bars
    plt.bar(r, orangeBars, bottom=greenBars, color='#f9bc86', edgecolor='white', width=barWidth, label="ictal")
    # Create blue Bars
    plt.bar(r, blueBars, bottom=[i+j for i,j in zip(greenBars, orangeBars)], 
            color='#a3acff', edgecolor='white', width=barWidth, label="interictal")
     
    # Custom x axis
    plt.xticks(r, names)
    plt.xlabel("Subject")
    plt.ylabel("Percent")
    
    # Add a legend
    plt.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    
    # Show graphic
    plt.show()


#%% plot for  correlations and eigen values
    
