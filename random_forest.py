# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 20:26:43 2017

@author: Armand

Random Forest Classifier
"""

from sklearn.ensemble import RandomForestClassifier
from train_test_split import train_test_split_data
from scoring_methods import r2_scorer
from scoring_methods import score_classifier_auc
import pickle
from sklearn.model_selection import GridSearchCV
from helper_functions import load_test_samples
from helper_functions import predictions_to_csv
from scoring_methods import armand_auc

#feature_folder="C:/Users/Armand-Laptop/Dropbox/seizure onset project/pickle data"
feature_folder="C:\Users\Armand\Dropbox\seizure onset project\pickle data"
#feature_folder="/media/fr3shc00kie/9E4CE8E74CE8BAE3/Users/Armand/Dropbox/seizure onset project/pickle data/"
rand_state=1
all_subjects=["Dog_1","Dog_2","Dog_3","Dog_4","Patient_1","Patient_2","Patient_3","Patient_4","Patient_5",
              "Patient_6","Patient_7","Patient_8"]
all_dfs=list()
for subject in all_subjects:    
    print 'Current Subject:', subject
    learner=RandomForestClassifier(n_estimators=3000,random_state=rand_state,bootstrap=False,min_samples_split=2)
    x_train,x_test,y_train,y_test=train_test_split_data(feature_folder,subject,perc_train=0.9)
    learner.fit(x_train,y_train)
    learner_prediction=learner.predict(x_test)
    r2_scorer(learner_prediction,y_test)
    
    
    #%% gridsearch info
    parameters={'n_estimators':[1000,3000,5000],'min_samples_split':[2,3,5,10], 'bootstrap':[False,True]}
    clf=GridSearchCV(learner,parameters)
    clf.fit(x_train,y_train)
    best_parameters=clf.best_params_
    print best_parameters
    learner_opt=RandomForestClassifier(n_estimators=best_parameters['n_estimators'],
                                   min_samples_split=best_parameters['min_samples_split'],random_state=1
                                   ,bootstrap=best_parameters['bootstrap'])
    
    learner_opt.fit(x_train,y_train)
    best_prediction=learner_opt.predict(x_test)
    r2_scorer(best_prediction,y_test)
    y_classes=[0,1,2]
    #%%

    S,E=score_classifier_auc(learner_opt, x_test,y_test,y_classes)
    score=0.5*(S+E)
    print 'Found best Parameters with score',score    
      #%%

    print "Armand Method of scoring score is:", armand_auc(learner_opt,x_test,y_test)
      
    #%% save classifier
    pickle_out_name=subject+'random_forest_learner.pickle'
    pickle_out=open(pickle_out_name,'wb')
    pickle.dump(learner_opt,pickle_out)
    pickle_out.close()
    
    #%% execute learner on all actual test data
    xtest,filenames=load_test_samples(subject,feature_folder)
    predictions=learner_opt.predict(xtest)
    df=predictions_to_csv(filenames,predictions,feature_folder)
    all_dfs.append(df)
#%%
import pandas
final_df=pandas.concat(all_dfs, ignore_index=True)

print "after dropping errors, size of final_df:",len(final_df['clip'])

#name fixing
ind=0
for name in final_df['clip']:
    if 'g_' in name:
        final_df['clip'][ind]='Do'+name
#save csv file
final_df.to_csv(path_or_buf='final_results.csv',index=False)