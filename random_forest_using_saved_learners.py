# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 08:33:12 2017

@author: Armand
run on saved learner

"""
from sklearn.ensemble import RandomForestClassifier
from train_test_split import train_test_split_data
from scoring_methods import r2_scorer
from scoring_methods import score_classifier_auc
#feature_folder="C:/Users/Armand-Laptop/Dropbox/seizure onset project/pickle data"
feature_folder="C:\Users\Armand\Dropbox\seizure onset project\pickle data"
home_dir="C:\Users\Armand\Dropbox\seizure detection project"
rand_state=1
all_subjects=["Dog_1","Dog_2","Dog_3","Dog_4","Patient_1","Patient_2","Patient_3","Patient_4","Patient_5",
              "Patient_6","Patient_7","Patient_8"]
all_dfs=list()

for subject in all_subjects:
    import pickle
    print 'Current subject: ' , subject
    pickle_in=subject+'random_forest_learner.pickle'
    learner_opt=pickle.load(open(pickle_in,"rb"))
    
    #%% execute learner on all actual test data
    
    #run only on Dog_1 for now

    from helper_functions import load_test_samples
    xtest,filenames=load_test_samples(subject,feature_folder)
    predictions=learner_opt.predict(xtest)
    '''
    y classes= 0, 1 , 2
    0=no seizure
    1=seizure
    2=early in seizure
    
    '''
    from helper_functions import predictions_to_csv
    df=predictions_to_csv(filenames,predictions,feature_folder)
    all_dfs.append(df)
#%%
import pandas
final_df=pandas.concat(all_dfs, ignore_index=True)

#checking for duplicate files
print "before error name check, size of final_df:",len(final_df['clips'])
error_names=list()
error_indices=[]
index=0
for name in final_df['clips']:
    if '(' in name:
        error_names.append(name)
        error_indices.append(index)
    index=index+1
final_df=final_df.drop(error_indices)
print "after dropping errors, size of final_df:",len(final_df['clips'])

#name fixing
ind=0
for name in final_df['clips']:
    if 'g_' in name:
        final_df['clips'][ind]='Do'+name
    ind=ind+1
#save csv file
final_df.to_csv(path_or_buf='final_results.csv',index=False)
