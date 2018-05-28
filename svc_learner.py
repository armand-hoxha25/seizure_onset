# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 18:54:40 2017

@author: Armand-Laptop
implement SVC as learner

"""
from train_test_split import train_test_split_data
from sklearn.metrics import r2_score
from sklearn.svm import SVC
rand_state=1

def r2_scorer(prediction,y_test):
    r2score=r2_score(y_test,prediction)
    print r2score

subject_list=['Dog_1','Dog_2','Dog_3','Dog_4','Dog_5','Patient_1',
              'Patient_2','Patient_3','Patient_4','Patient_5','Patient_6,',
              'Patient_7','Patient_8']

for subject in subject_list:
    print subject
    learner=SVC(random_state=rand_state)
    feature_folder='C:\Users\Armand\Google Drive\seizure detection project\pickle data'
    x_train,x_test,y_train,y_test=train_test_split_data(feature_folder=feature_folder,subject='Dog_2',perc_train=0.7)
    learner.fit(x_train,y_train)
    prediction=learner.predict(x_test)
    r2_scorer(prediction,y_test)
    
    #%% gridsearch info
    parameters={'kernel':('linear','rbf','poly','sigmoid'),'C':range(1,25),'degree':range(3,20),'probability':[False,True]}
    from sklearn.model_selection import GridSearchCV
    clf=GridSearchCV(learner,parameters)
    clf.fit(x_train,y_train)
    best_parameters=clf.best_params_
    
    #%%
    
    learner=SVC(kernel=best_parameters['kernel'],C=best_parameters['C'],degree=best_parameters['degree'],random_state=rand_state)
    learner.fit(x_train,y_train)
    best_prediction=learner.predict(x_test)
    r2_scorer(best_prediction,y_test)
    

#%% run learner on all data
