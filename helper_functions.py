# -*- coding: utf-8 -*-
"""
Spyder Editor
author: Armand Hoxha

Helper functions

"""

def load_test_samples(subject,feature_folder):
    import glob
    import pickle
    from train_test_split import unpickle_feature
    all_subject_files=glob.glob(feature_folder+"/"+subject+"*")
    xtest=list()
    filenames=list()
    for filename in all_subject_files:
        if filename.find('test')is not -1 and filename.find("(") is -1:
            feature=unpickle_feature(filename)
            xtest.append(feature)
            filenames.append(filename)
    return xtest, filenames

def fix_filenames(filenames,feature_folder):
    new_list=list()
    for filename in filenames:
        temp_name=filename.strip(feature_folder)
        temp_name=temp_name.strip("\ ")
        temp_name=temp_name.strip('.pickle')
        new_list.append(temp_name)
    return new_list
    
def predictions_to_csv(filenames,predictions,feature_folder):
    #columns : clip, seizure,early
    import pandas
    seizure_results=list()
    early_results=list()
    for prediction in predictions:
        early_results.append(prediction[0])
        seizure_results.append(prediction[1])

#    print len(seizure_results) 
#    print len(early_results)  
#    print len(filenames)                  
    final_dict={'clip':fix_filenames(filenames,feature_folder),'early':early_results,'seizure':seizure_results}
    df=pandas.DataFrame(final_dict)
    df.to_csv(path_or_buf='results.csv',index=False)
    return df