import pandas as pd
import re
import nltk
import psycopg2
import sys
import pickle
from nltk.tag.perceptron import PerceptronTagger
import pandas as pd
import numpy as np



tagger = PerceptronTagger()
from nltk.tokenize import word_tokenize

import os
#need to fix

from pos_phrases import PosPhrases
phraser = PosPhrases()
voc = pd.read_pickle('/model_input/phraserloglog-vocab.pkl')

phraser.vocab = voc

tfidf = pickle.load(open('/model_input/tfidf_both_50000.pickle','rb'))

sentences_all = []
phrases_all = []
volids_all = []
state_all = []
year_all = []


chi_square_selector = pickle.load(open('/Logistic/chi2_selector_tfidf_2000.pickle','rb'))
best_model = pickle.load(open('/Logistic/best_model_calibrated_2000.pickle','rb'))


df = pickle.load(open('/data/df_sample1000.pickle','rb'))

  
 for i in range(0,len(df)):
     statutes = df.iloc[i]['statutes']
     for sent in statutes:
         if len(sent)>10:
             s = re.findall(r"[a-z]+-?[a-z]+", sent.lower(),flags = re.UNICODE)
             words = [w for i,w in enumerate(s) if s[i] != s[i-1]]
             phrases = phraser.phrase(words, ignore_POS = True) 
             w=' '
             if phrases is not None:
                 if len(phrases)>10:
                     for j in phrases:
                         w+=j+' '
                     sentences_all.append(sent)
                     phrases_all.append(w)

            
tfidf_X = tfidf.transform(phrases_all)  
X = chi_square_selector.transform(tfidf_X)
prediction_all = best_model.predict_proba(X)[:,1]

df_all = pd.DataFrame()
df_all['Sentences'] = sentences_all
df_all['Phrases'] = phrases_all
df_all['prediction'] = prediction_all


df_all.to_csv('df_sentences.csv')
