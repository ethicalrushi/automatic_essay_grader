#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 08:21:26 2019

@author: rushikesh
"""

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from utils import  extract_features
import pickle

#importing training dataset
dataset = pd.read_table('data/training_set_rel3.tsv', encoding = "ISO-8859-1")

#Training tf-idf vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
vect = TfidfVectorizer(ngram_range=(1,4)).fit(dataset['essay'])
vect.get_feature_names()
len(vect.get_feature_names())
print("Tf-idf vectorizer built successfully")

#Picking vect
save_vect = open("vect.pickle","wb")
pickle.dump(vect, save_vect)
save_vect.close()


dataset2 = dataset[dataset['essay_set']==2]
dataset2 = dataset2[['essay_id','essay_set','essay','domain1_score','domain2_score']]
 
#Creating dataset for set 3
dataset_train3 = dataset[['essay_id','essay_set','essay','domain1_score']]
dataset_train3 = dataset_train3[dataset_train3['essay_set']!=2]

X_train = extract_features(dataset_train3, vect, 'data/prompt1.txt','data/prompt1_quesn.txt')
X_train = pd.DataFrame(data=X_train)
X_train.columns =['relevance','relevance_quesn','essay_set','word_count','distinct_word_count','sentence_count','avg_word_length']

#for set2
X_train2 = extract_features(dataset2, vect, 'data/prompt2.txt', 'data/prompt2_quesn.txt')
X_train2 = pd.DataFrame(data=X_train2)
X_train2.columns = ['relevance','relevance_quesn','essay_set','word_count','distinct_word_count','sentence_count','avg_word_length']

print("Extracted training features successfully")

#Scaling features

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaled_X_train = scaler.fit_transform(X_train)

#for set 2
scaled_X_train2 = scaler.fit_transform(X_train2)


#Fitting models

from sklearn.svm import SVC
from sklearn import svm

#for set 1,3,4,5,6,7,8
classifier = SVC( C=10,gamma=0.15)
classifier.fit(scaled_X_train,dataset_train3['domain1_score'])  #Requires scaling the feature

#for domain 1 of set2
classifier2_a = svm.SVR(C=10, gamma=0.01)
classifier2_a.fit(scaled_X_train2,dataset2['domain1_score']) #Requires scaling the features

#fir domain 2 of set 2
classifier2_b = svm.SVR(C=10, gamma=0.01)
classifier2_b.fit(scaled_X_train2,dataset2['domain2_score'])  #Requires scaling the features

print("models trained successfully")


#Picking models
save_model = open("classifier.pickle","wb")
pickle.dump(classifier, save_model)
save_model.close()

save_model = open("classifier2_a.pickle","wb")
pickle.dump(classifier2_a, save_model)
save_model.close()

save_model = open("classifier2_b.pickle","wb")
pickle.dump(classifier2_b, save_model)
save_model.close()
