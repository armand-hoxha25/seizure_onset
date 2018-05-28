
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 20:26:43 2017

@author: Armand

adaboost classifier with no optimization and my AUC scoring method

"""

from sklearn.ensemble import AdaBoostClassifier
from train_test_split import train_test_split_data
#from scoring_methods import r2_scorer
#from scoring_methods import score_classifier_auc
from scoring_methods import armand_auc
#feature_folder="C:/Users/Armand-Laptop/Dropbox/seizure onset project/pickle data"
#feature_folder="C:\Users\Armand\Dropbox\seizure onset project\pickle data"
#feature_folder="/media/fr3shc00kie/9E4CE8E74CE8BAE3/Users/Armand/Dropbox/seizure onset project/pickle data/"
#feature_folder="D:\documents\ICG penn state seizure\clips\pickle data"
feature_folder="/home/fr3shc00kie/Desktop/seizure onset project/pickle data/"
rand_state=1
all_subjects=["Dog_1","Dog_2","Dog_3","Dog_4","Patient_1","Patient_2","Patient_3","Patient_4","Patient_5",
              "Patient_6","Patient_7","Patient_8"]

all_dfs=list()
for subject in all_subjects:    
    print 'Current Subject:', subject
    learner=AdaBoostClassifier(n_estimators=1000,random_state=rand_state)
    x_train,x_test,y_train,y_test=train_test_split_data(feature_folder,subject,perc_train=0.9)
    learner.fit(x_train,y_train)
    learner_prediction=learner.predict(x_test)
    #%%

    print "Armand Method of scoring score is:", armand_auc(learner,x_test,y_test)
    
    #%% save classifier
    import pickle
    pickle_out_name=subject+'random_forest_learner.pickle'
    pickle_out=open(pickle_out_name,'wb')
    pickle.dump(learner,pickle_out)
    pickle_out.close()
    
    #%% execute learner on all actual test data
    
    #run only on Dog_1 for now

    from helper_functions import load_test_samples
    xtest,filenames=load_test_samples(subject,feature_folder)
    predictions=learner.predict_proba(xtest)
 
    from helper_functions import predictions_to_csv
    df=predictions_to_csv(filenames,predictions,feature_folder)
    all_dfs.append(df)
#%%
import pandas
final_df=pandas.concat(all_dfs, ignore_index=True)

#name fixing
ind=0
for name in final_df['clip']:
    if 'g_' in name:
        final_df['clip'][ind]='Do'+name
    ind=ind+1
    #%%
#save csv file
final_df.to_csv(path_or_buf='final_results.csv',index=False)