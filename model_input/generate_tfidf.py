import pandas as pd
import numpy as np
import itertools
import os
import re
import logging 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split  
import joblib 
from joblib import Parallel, delayed

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC


from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score
from sklearn.calibration import calibration_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
from sklearn.metrics import confusion_matrix
from sklearn.calibration import calibration_curve
import pickle

from collections import Counter
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import gc 
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.feature_extraction import text
os.chdir('/cluster/work/lawecon/Work/lhan/tax_related/')

logging.basicConfig(format='%(asctime)s  %(message)s', 
    datefmt='%y-%m-%d %H:%M:%S', level=logging.INFO)

stop_words = text.ENGLISH_STOP_WORDS
us_states =  ['alabama', 'alaska','arizona', 'arkansas', 'california', 'colorado', 'connecticut', 'delaware', 'florida', 'georgia', 'hawaii', 'idaho', 'illinoi', 'indiana', 'iowa', 'kansas','kentucky','louisiana', 'maine', 'maryland', 'massachusetts', 'michigan', 'minnesota', 'mississippi', 'missouri', 'montana',     'nebraska', 'nevada', 'new_hampshire', 'new_jersey', 'new_mexico', 'new_york', 'north_carolina', 'north_dakota', 'ohio', 'oklahoma', 'oregon', 'pennsylvania', 
   'rhode_island', 'south_carolina', 'south_dakota', 'tennessee', 'texas', 'utah', 'vermont', 'virgnia', 'washington', 'west_virginia', 'wisconsin', 'wyoming']  
us_cities = ['montgomery', 'juneau', 'phoenix', 'little_rock', 'sacramento', 'denver', 'harford', 'dover', 'tallahassee', 'atlanta', 'boise', 'springfield', 'indianapolis', 'des_moines', 'topeka', 'frankfort', 'baton_rouge', 'augusta', 'annapolis', 'boston', 'lansing', 'saint_paul', 'jackson', 'jefferson_city', 'helena', 'lincoln', 'carson_city', 'concord', 'trenton', 'santa_fe', 'albany', 'raleigh', 'bismarck', 'columbus', 'oklahoma_city', 'salem', 'harrisburg', 'providence', 'columbia', 'pirre', 'nashville', 'austin', 'salt_lake_city', 'montpelier', 'richmond', 'olympia', 'charleston', 'madison', 'cheyenne']


    
phrase2int = pd.read_pickle('/cluster/work/lawecon/Work/state_laws/data/tax_law_related_data/2-clean/dictionaries/statutes-phrase2int.pkl')              
phrase_dict = {value:key for key,value in phrase2int.items()}



phrase_state= pickle.load(open('/cluster/work/lawecon/Work/lhan/tax_related/phrase_half.pickle','rb'))
data_lexis = pickle.load(open('/cluster/work/lawecon/Work/lhan/tax_related/df_lexis.pickle','rb'))

logging.info(str(len(phrase_state)))

phrase_lexis=[i for i in data_lexis.phrases]

phrase_all = phrase_lexis + phrase_state

phrase_lst = []

for i in phrase_all:
  w=' '
  for j in i:
    w += phrase_dict[j] + ' '
  phrase_lst.append(w)
  


tfidf = TfidfVectorizer(max_features=50000,max_df=0.75, stop_words = list(stop_words)+us_states+us_cities)

X = tfidf.fit_transform(phrase_lst)
pickle.dump(tfidf, open("tfidf_both_50000.pickle", "wb"))