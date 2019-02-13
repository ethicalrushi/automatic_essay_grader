# automatic_essay_grader

Trained a model to automatically grade the essays available on this dataset
https://www.kaggle.com/c/asap-aes

Instructions to test the results:
1) Install the requirements.txt
2) Train the model by running the train.py file
3) Test the model by running the test.py file(You can add the test_set results in this file as mentioned in the comments.

<b>Features considered:</b><br>
i) Tf-idf matrix(tf-idf)<br>
ii) Relevance with the source essays(relevance)<br>
iii) Relevance with the prompt(relevance_quesn)<br>
iv) Number of words used(word_count)<br>
v) Number of distinct words used(distinct_word_count)<br>
vi) Number of sentences (sentence_count)<br>
vii) Average word sixe (avg_word_length)<br>
viii) Essay set(essay_set)
<br>
(Note: Grammatical mistakes and spelling mistakes are ignored since they were not considered while grading as mentioned in the scoring docs.)
Following graph shows the kappa score on essay set1 for dfferent featuresets.
![Alt text](https://github.com/ethicalrushi/automatic_essay_grader/blob/master/featureset_set1.png)

featureset1: tfidf with unigrams, relevance, relevance_quesn, essay_set<br>
featureset2: relevance, relevance_quesn, word_count, distinct_word_count,essay_set, sentence_count, avg_word_count<br>
featureset3: relevance, relevance_quesn, word_count, distinct_word_count,essay_set, sentence_count,                   avg_word_count(considering 1,2,3,4 grams)<br>

Thus featureset3 looks most promising and is used for all the models.<br>

The data is divided into two sets viz. set a and set b.<br>
Set a contains essaysets 1,3,4,5,6,7,8 and Set b contains essay set 2.<br>
This division is made since the essays in set 2 have 2 predictions to be made i.e domain1 and domain2 scores.<br>

Following graph shows the feature importance of features as evaluated on Set a<br>

![Alt text](https://github.com/ethicalrushi/automatic_essay_grader/blob/master/feature_importance.png)

<b>Model Selection:</b><br>
(a) For set A:
Following graph shows the kappa scores of various models trained and tested on essay set 1<br>

![Alt text](https://github.com/ethicalrushi/automatic_essay_grader/blob/master/models_set1.png)

And this is the score variation on setA(i,e essay set 1,3,4,5,6,7,8)
![Alt text](https://github.com/ethicalrushi/automatic_essay_grader/blob/master/models_seta.png)

(Note: The increase in score is due to the essay_set features whose importance was null when trained on essay sets of single type since the value of essay_set for all data points in such case is same but it got a high importance when trained on all essay_sets containing data points with different essay_set values.)

Owing to the scores in above graph SVC is used as the final model for set A.

(b) For set B:
Models evaluated- Linear regression, SVR
Following graph shows the kappa scores for domain 1 of various models trained and tested on essay set B<br>
![Alt text](https://github.com/ethicalrushi/automatic_essay_grader/blob/master/models_set2_domain1.png)

Following graph shows the kappa scores for domain 2 of various models trained and tested on essay set B<br>
![Alt text](https://github.com/ethicalrushi/automatic_essay_grader/blob/master/models_set2_domain2.png)

Owing to the results above SVR was used as the final model for set B.

Final scores on individual sets:<br>
![Alt text](https://github.com/ethicalrushi/automatic_essay_grader/blob/master/different_sets.png)



