#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 10:17:47 2019

@author: rushikesh
"""
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import TweetTokenizer
from sklearn.metrics.pairwise import cosine_similarity


def extract_prompt(filename1,filename2,vect):
    prompt1 = open(filename1).read()
    
    prm1 = [s.strip() for s in prompt1.splitlines() if len(s.strip())>0 ]
    prompt2 = open(filename2).read()
    prm1_quesn = [s.strip() for s in prompt2.splitlines() if len(s.strip())>0 ]
    
    #Converting prm1 to a single string so that similarity is found across the whole prompt rather than five sent individually
    prm1 = "\n".join(s for s in prm1)
    prm1_quesn = "\n".join(s for s in prm1_quesn)
    #Vetorizing prompts
    prm1_vect = vect.transform([prm1])
    prm1_quesn_vect = vect.transform([prm1_quesn,])
    
    return prm1_vect, prm1_quesn_vect

""" 
Since dataframemapper or sklearn pipeline can't be just transformed, it has to be fitted prior.
We don't want to fit vect on the train_data of set1 again since we want the vect fitted on entire corpus.
Hence we create a custom transformer which does nothinng for fit and just transforms the text.
"""
from sklearn.base import TransformerMixin
class Vectorize(TransformerMixin, object): #inheriting from ransformerMixin allows us to use fit_transform and some other defaults
    
    def __init__(self, vect):
        self.vect = vect
    def transform(self,X):
        #print(type(X))
        vecty = self.vect
        #print(len(vecty.get_feature_names()))
        trans = vecty.transform(X)
        #print(trans.shape[1])
        return trans.astype(float)
    
    def fit(self,X,y=None):
        #print(type(X))
        return self #we don't want to fit vect on this


def extract_features(dataset1, vect, prom_filename1, prom_filename2):
    #Adding 3 more features tf-idf_vector and relevance  
    dataset1['relevance'] = np.nan
    dataset1['relevance_quesn'] = np.nan
    dataset1['word_count'] = np.nan
    #Adding tf-idf vectors as features
    #currently set to [essay] , to be transformed using mapper
    #making a list since tf-idf needs an iterable over text
    dataset1['tf-idf'] = dataset1['essay']
    tokenizer_words = TweetTokenizer()
    dataset1['word_count'] = list(map(len,dataset1['essay'].apply(tokenizer_words.tokenize)))
    dataset1['distinct_word_count'] = list(map(len,list(map(set,dataset1['essay'].apply(tokenizer_words.tokenize)))))
    dataset1['sentence_count'] = list(map(len,dataset1['essay'].apply(nltk.sent_tokenize)))
    dataset1['charslength'] = list(map(len,list(map("".join,dataset1['essay'].apply(tokenizer_words.tokenize)))))
    dataset1['avg_word_length'] = np.divide(dataset1['charslength'],dataset1['word_count'])
  
    #Calculating similarity between prompt and essay and adding a feature relevance
    
    
    prm1_vect, prm1_quesn_vect = extract_prompt(prom_filename1, prom_filename2,vect)
    
    essay_vect = vect.transform(dataset1['essay'])
    sim1 = cosine_similarity(essay_vect, prm1_vect) #don't compute element wise , broadcasting reduces time
    sim2 = cosine_similarity(essay_vect, prm1_quesn_vect)
    dataset1['relevance']= sim1
    dataset1['relevance_quesn'] = sim2
   
   #note- Dataframemapper gives nonetype not iterable error if is not fitted to data.
   #It is necessary to fit mapper before transforming, otherwise it recieves no data and 
   #hence the nonetype error.
    
    from sklearn_pandas import DataFrameMapper
    mapper = DataFrameMapper([
        # ('tf-idf', Vectorize(vect),{'input_df': True}), #transforming tf-idf using vect
         ('relevance', None), #no transformation required
         ('relevance_quesn',None), #no transformation required
         ('essay_set',None),
         ('word_count',None),
         ('distinct_word_count',None),
         ('sentence_count', None),
         ('avg_word_length',None),

         #if we include essay then it would cause error in final stacking of colums as 
         #tf-idf is sparse and text cannot be converted to sparse format 
         #Also essay text is not required
        
     ])
    #creating features combinining tf-idf and relevance features
    mapp = mapper.fit(dataset1) #applies individual column transformation as defined in mapper

    features = mapp.transform(dataset1)
    
    return features

def build_results(class_pred, class1_pred2_dom1, class1_pred2_dom2, valid2):
    pred_2 = np.zeros(len(class1_pred2_dom1)+ len(class1_pred2_dom2), dtype=class1_pred2_dom2.dtype)
    pred_2[0::2] = class1_pred2_dom1
    pred_2[1::2] = class1_pred2_dom2
    val_len = len(class_pred)+len(pred_2)
    ind = valid2.index.tolist()
    ind1 = ind[0]
    ind2 = ind[-1]
    ind3 = ind2+ind2-ind1+2
    validn_res = np.zeros(val_len,dtype=class_pred.dtype)
    validn_res[:ind1] = class_pred[:ind1] #set 1 upto 588 index
    validn_res[ind1:ind3] = pred_2[:]
    validn_res[ind3:] = class_pred[ind1:]
    return validn_res

def build_comparison(pred_res, true_res):
    true_res = list(true_res)
    pred_res = list(pred_res)
    frames = [ pd.DataFrame(data=pred_res),pd.DataFrame(data=true_res)]
    comparison = pd.concat(frames, axis=1)
    comparison.columns = ['Predicted Result','True Result']
    return comparison
