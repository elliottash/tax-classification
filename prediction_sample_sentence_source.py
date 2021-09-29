import pandas as pd
import re
import nltk
import sys
import pickle
from nltk.tag.perceptron import PerceptronTagger
import pandas as pd
import numpy as np

#address fixed


tagger = PerceptronTagger()
from nltk.tokenize import word_tokenize

import os

from pos_phrases import PosPhrases
phraser = PosPhrases()
voc = pd.read_pickle('/model_input/phraserloglog-vocab.pkl')

phraser.vocab = voc


prediction_path = '/Prediction'
os.chdir(prediction_path)
tfidf = pickle.load(open('/model_input/tfidf_both_50000.pickle','rb'))
chi_square_selector = pickle.load(open('/tax_source/chi2_selector_tfidf_30000.pickle','rb'))
best_model = pickle.load(open('/tax_source/best_model_tfidf_30000.pickle','rb'))
phrase2int = pd.read_pickle('/model_input/statutes-phrase2int.pkl')         
phrase_dict = {value:key for key,value in phrase2int.items()}

sentences_all = []
phrases_all = []

other_all= []
corporate_tax = []
energy_tax = []
excise_tax = []
income_tax = []
inheritance_tax = []
license_tax = []
property_tax = []
sales_tax = []

class_mapping = {0:other_all,1:corporate_tax,2:energy_tax, 3:excise_tax, 4:income_tax, 5:inheritance_tax, 6:license_tax, 7:property_tax, 8:sales_tax}


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
prediction_all = best_model.predict(X)

for i in list(class_mapping.keys()):
  class_mapping[i] = class_mapping[i] +list(best_model.predict_proba(X)[:,i]) 


df_all = pd.DataFrame()
df_all['Sentences'] = sentences_all
df_all['Phrases'] = phrases_all
df_all['prediction'] = prediction_all
df_all['Other'] = class_mapping[0]
df_all['corporate_tax'] = class_mapping[1]
df_all['energy_tax'] = class_mapping[2]
df_all['exise_tax'] = class_mapping[3]
df_all['income_tax'] = class_mapping[4]
df_all['inheritance_tax'] = class_mapping[5]
df_all['license_tax'] = class_mapping[6]
df_all['property_tax'] = class_mapping[7]
df_all['sales_tax'] = class_mapping[8]

  

df_all.to_csv('/Prediction/df_sentences_source.csv')