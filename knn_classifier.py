# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 20:26:43 2017

@author: Armand

Random Forest Classifier with no optimization and my AUC scoring method

"""

from sklearn.ensemble import RandomForestClassifier
from sklearn import neighbors
from train_test_split import train_test_split_data
#from scoring_methods import r2_scorer
#from scoring_methods import score_classifier_auc
from scoring_methods import armand_auc
from sklearn.grid_search import GridSearchCV
feature_folder="/media/fr3shc00kie/9E4CE8E74CE8BAE3/Users/Armand/Dropbox/seizure onset project/pickle data/"
rand_state=1
all_subjects=["Dog_1","Dog_2","Dog_3","Dog_4","Patient_1","Patient_2","Patient_3","Patient_4","Patient_5",
              "Patient_6","Patient_7","Patient_8"]

all_dfs=list()
for subject in all_subjects:    
    print 'Current Subject:', subject
    learner=neighbors.KNeighborsClassifier(3,weights='distance')
    x_train,x_test,y_train,y_test=train_test_split_data(feature_folder,subject,perc_train=0.9)
    learner.fit(x_train,y_train)
    learner_prediction=learner.predict(x_test)
    #%%
    print "Armand Method of scoring score is:", armand_auc(learner,x_test,y_test)
    #%% optimize learner
    parameters={'n_neighbors':[3,5,10,15],'weights':['distance','uniform'],'algorithm':['kd_tree','ball_tree'],'leaf_size':[30,50,100]}
    clf=GridSearchCV(learner,parameters,scoring=armand_auc)
    clf.fit(x_train,y_train)
    best_parameters=clf.best_params_
    print best_parameters
    learner_opt=neighbors.KNeighborsClassifier(n_neighbors=best_parameters['n_neighbors'],
                                   weights=best_parameters['weights']
                                   ,algorithm=best_parameters['algorithm'],leaf_size=best_parameters['leaf_size'])
    learner_opt.fit(x_train,y_train)
    #%%
    print "Optimized learner score:", armand_auc(learner_opt,x_test,y_test)
    #%% save classifier
    import pickle
    pickle_out_name=subject+'random_forest_learner.pickle'
    pickle_out=open(pickle_out_name,'wb')
    pickle.dump(learner,pickle_out)
    pickle_out.close()
    
    #%% execute learner on all actual test data
    learner=learner_opt
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