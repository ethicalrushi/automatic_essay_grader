# Automatic_Essay_Grader

Trained a model to automatically grade the essays available on this dataset
https://www.kaggle.com/c/asap-aes

<b>Instructions to test the results:</b>
1) Train the model by running the train.py file
2) Test the model by running the test.py file(You can add the test_set results in this file as mentioned in the comments of test.py and evaluate the test set accuracy).

<b>Features considered:</b><br>
i) Tf-idf matrix(tf-idf)<br>
ii) Relevance with the source essays(relevance)<br>
iii) Relevance with the prompt(relevance_quesn)<br>
iv) Number of words used(word_count)<br>
v) Number of distinct words used(distinct_word_count)<br>
vi) Number of sentences (sentence_count)<br>
vii) Average word size (avg_word_length)<br>
viii) Essay set(essay_set)
<br>
(Note: Grammatical mistakes and spelling mistakes are ignored since they were not considered while grading as mentioned in the scoring docs).
<br><br><b>
Following graph shows the kappa score on essay set1 for dfferent featuresets.</b>
![Alt text](https://github.com/ethicalrushi/automatic_essay_grader/blob/master/featureset_set1.png)

featureset1: tfidf with unigrams, relevance, relevance_quesn, essay_set<br>
featureset2: relevance, relevance_quesn, word_count, distinct_word_count,essay_set, sentence_count, avg_word_length<br>
featureset3: relevance, relevance_quesn, word_count, distinct_word_count,essay_set, sentence_count,                   avg_word_length(considering 1,2,3,4 grams)<br>

Thus featureset3 looks most promising and is used for all the models.<br>

The data is divided into two sets viz. set a and set b.<br>
Set a contains essaysets 1,3,4,5,6,7,8 and Set b contains essay set 2.<br>
This division is made since the essays in set 2 have 2 predictions to be made i.e domain1 and domain2 scores.<br>

<b>Following graph shows the feature importance of features as evaluated on Set a</b><br>

![Alt text](https://github.com/ethicalrushi/automatic_essay_grader/blob/master/feature_importance.png)

<b>Model Selection:</b><br>
<b>(a) For set A:</b>
<br>
Following graph shows the kappa scores of various models trained and tested on essay set 1<br>

![Alt text](https://github.com/ethicalrushi/automatic_essay_grader/blob/master/models_set1.png)

And this is the score variation on setA(i,e essay set 1,3,4,5,6,7,8)
![Alt text](https://github.com/ethicalrushi/automatic_essay_grader/blob/master/models_seta.png)

(Note: The increase in score is due to the essay_set features whose importance was null when trained on essay sets of single type since the value of essay_set for all data points in such case is same but it got a high importance when trained on all essay_sets containing data points with different essay_set values.)

Owing to the scores in above graph <b>SVC is used as the final model for set A.<b>

<b>(b) For set B: 
</b>
Models evaluated- Linear regression, SVR
<br>
Following graph shows the kappa scores for domain 1 of various models trained and tested on essay set B<br>
![Alt text](https://github.com/ethicalrushi/automatic_essay_grader/blob/master/models_set2_domain1.png)

Following graph shows the kappa scores for domain 2 of various models trained and tested on essay set B<br>
![Alt text](https://github.com/ethicalrushi/automatic_essay_grader/blob/master/models_set2_domain2.png)

Owing to the results above <b>SVR was used as the final model for set B.</b>

<b>Final scores on individual sets:</b><br>
![Alt text](https://github.com/ethicalrushi/automatic_essay_grader/blob/master/different_sets.png)

<b>Final scores on overall validation data:</b><br>
![Alt text](https://github.com/ethicalrushi/automatic_essay_grader/blob/master/final_score.png)

<b>Final kappa score on validation data :0.9853 
 </b>
(Note: All the scores are calculated on the validation data since true labels of test data were not available)

 <b>Notes:(for extending the model to grade essays written by more matured writers)
 </b><br>
i) Features like visual nature( can be calculated using British Natural Corpus), beautiful words(using Cornell Math Cryptography) and emotive_effectiveness(using MPQA) can be used to score essays written by more matured writers(for e.g. Pulitzer prize essays). Since the essays are written by school kids of grade 8,9,10th on a very short notice these features are not that considerable.<br>
Reference:https://nlp.stanford.edu/courses/cs224n/2013/reports/song.pdf <br>
ii) Spelling errors can be calculated using English words from Python’s NLTK
iii) Lexical diversity can be accounted for using the ratio of the count of all lexically important POS tags such as
nouns, adjectives and adverbs to the count of all tags.<br>




