#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 08:33:14 2019

@author: rushikesh
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pickle 
from utils import extract_features, build_results, build_comparison
from sklearn.preprocessing import StandardScaler

classifier_f=open("classifier.pickle","rb")
classifier= pickle.load(classifier_f)
classifier_f.close()

classifier_f=open("classifier2_a.pickle","rb")
classifier2_a= pickle.load(classifier_f)
classifier_f.close()

classifier_f=open("classifier2_b.pickle","rb")
classifier2_b= pickle.load(classifier_f)
classifier_f.close()

classifier_f=open("vect.pickle","rb")
vect= pickle.load(classifier_f)
classifier_f.close()

print("Models and vectorizers imported successfully")
#Test function
def test_results(test_file,lower,upper, classifier2_a,classifier2_b,vect):
    valid = test_file.iloc[:,:]
    valid3 = valid[valid['essay_set']!=2]
    valid2 = valid[valid['essay_set']==2]
    
    X_test = extract_features(valid3, vect, 'data/prompt1.txt', 'data/prompt1_quesn.txt') #all sets except set 2
    X_test = pd.DataFrame(data= X_test)
    X_test.columns =['relevance','relevance_quesn','essay_set','word_count','distinct_word_count','sentence_count','avg_word_length']
    
    #for set 2
    X_test2 = extract_features(valid2, vect, 'data/prompt2.txt', 'data/prompt2_quesn.txt')
    X_test2 = pd.DataFrame(data=X_test2)
    X_test2.columns = ['relevance','relevance_quesn','essay_set','word_count','distinct_word_count','sentence_count','avg_word_length']
    
    #Scaling features
    
    scaler = StandardScaler()
    
    scaled_X_test = scaler.fit_transform(X_test)
    scaled_X_test2 = scaler.fit_transform(X_test2)
    
    #Predicting validation results
    
    #for set 1,3,4,5,6,7,8
    class_pred = classifier.predict(scaled_X_test)
    
    #for set 2
    class_pred2_dom1 = classifier2_a.predict(scaled_X_test2)
    
    #Rounding off and clipping for max and min range
    class_pred2_dom1= np.clip(np.round(class_pred2_dom1),lower, upper).astype(int)
    
    class_pred2_dom2 = classifier2_b.predict(scaled_X_test2)
    class_pred2_dom2= np.clip(np.round(class_pred2_dom2), np.min(true_res2), np.max(true_res2)).astype(int)
    
    validation_results = build_results(class_pred, class_pred2_dom1, class_pred2_dom2, valid2)
    

    return validation_results

#validation set
validation_set = pd.read_table('data/valid_set.tsv', encoding = "ISO-8859-1")
valid_results = pd.read_csv('data/valid_sample_submission_5_column.csv')
valid_results2 = valid_results[valid_results['essay_set']==2]
true_res2 = np.array(valid_results2['predicted_score'])
lower = np.min(true_res2) 
upper =  np.max(true_res2)
                            
#test set
test_set = pd.read_table('data/test_set.tsv',encoding = "ISO-8859-1") #true scores of validn set
"""
Add a test_results_file in data folder
It will be similar to valid_sample_submission_5
test_true_results = pd.read_csv('data/test_results_filename')
"""
#Predicting results
validation_results = test_results(validation_set,lower,upper, classifier2_a,classifier2_b,vect)
testset_results = test_results(test_set,lower,upper, classifier2_a,classifier2_b,vect)
print("Results ready")
#Comparing true results and prediction
comp_validation = build_comparison(validation_results,valid_results['predicted_score'])

"""
build a comparison table for test set.

comp_test = build_comparison(testset_results, test_true_results)
"""

#Kappa scores
from sklearn.metrics import cohen_kappa_score

#true scores for validation
validn_label = valid_results['predicted_score']

validation_score = cohen_kappa_score(validation_results,validn_label , weights="quadratic")
print("Validation_score:",validation_score)

"""
test_label = test_true_results['predicted_score']

test_set_score = cohen_kappa_score(testset_results,validn_label , weights="quadratic")
"""
