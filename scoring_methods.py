# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 20:28:57 2017
Collection of scoring methods
@author: Armand
scoring methods taken from Michael Hills github on seizure detection
"""

from sklearn.metrics import r2_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
import numpy

def r2_scorer(prediction,y_test):
    r2score=r2_score(y_test,prediction)
    print r2score

def translate_prediction(prediction, y_classes):
    if len(prediction) == 3:
        # S is 1.0 when ictal <=15 or >15
        # S is 0.0 when interictal is highest
        ictalLTE15, ictalGT15, interictal = prediction
        S = ictalLTE15 + ictalGT15
        E = ictalLTE15
        return S, E
    elif len(prediction) == 2:
        # 1.0 doesn't exist for Patient_4, i.e. there is no late seizure data
        if not numpy.any(y_classes == 1.0):
            ictalLTE15, interictal = prediction
            S = ictalLTE15
            E = ictalLTE15
            # y[i] = 0 # ictal <= 15
            # y[i] = 1 # ictal > 15
            # y[i] = 2 # interictal
            return S, E
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()

def score_classifier_auc(classifier, X_cv, y_cv, y_classes):
    predictions = classifier.predict_proba(X_cv)
    S_predictions = []
    E_predictions = []
    S_y_cv = [1.0 if (x == 0.0 or x == 1.0) else 0.0 for x in y_cv]
    E_y_cv = [1.0 if x == 0.0 else 0.0 for x in y_cv]

    for i in range(len(predictions)):
        p = predictions[i]
        S, E = translate_prediction(p, y_classes)
        S_predictions.append(S)
        E_predictions.append(E)

    fpr, tpr, thresholds = roc_curve(S_y_cv, S_predictions)
    S_roc_auc = auc(fpr, tpr)
    fpr, tpr, thresholds = roc_curve(E_y_cv, E_predictions)
    E_roc_auc = auc(fpr, tpr)

    return S_roc_auc, E_roc_auc

def armand_auc(classifier, x_test, y_test):
    # y[i] = 0 # ictal <= 15
    # y[i] = 1 # ictal > 15
    # y[i] = 2 # interictal
    predictions=classifier.predict(x_test)
    predictions=predictions.tolist()
    S_predictions=list()
    E_predictions=list()
    temp_s_test=list()
    temp_e_test=list()

    for index in range(len(predictions)-1):
        if y_test[index] is 1:
            S_predictions.append(predictions[index])
            temp_s_test.append(y_test[index])
        elif y_test[index] is 0:
            E_predictions.append(predictions[index])
            temp_e_test.append(y_test[index])
        elif y_test[index] is 2:
            temp_s_test.append(0)
            temp_e_test.append(1)            
            if predictions[index] is 2:
                E_predictions.append(1)
                if 1 in y_test:# because patient 4 has no mid-seizure data
                    S_predictions.append(0)
            else:
                E_predictions.append(0)#intentional wrong for purpose of scoring
                if 1 in y_test:
                    S_predictions.append(1)

            
    if len(S_predictions)>1:
        S=roc_auc_score(temp_s_test,S_predictions)  
        E=roc_auc_score(temp_e_test,E_predictions)
        final_score=(S+E)*0.5
    else:
        E=roc_auc_score(temp_e_test,E_predictions)
        final_score=E
    return final_score
    
    
