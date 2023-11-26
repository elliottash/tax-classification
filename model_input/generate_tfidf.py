
# Importing necessary libraries
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical operations
import itertools  # For efficient looping
import os  # For operating system related operations
import re  # For regular expression operations
import logging  # For logging information and debugging
from sklearn.feature_extraction.text import TfidfVectorizer  # For converting text to TF-IDF feature matrix
from sklearn.model_selection import train_test_split  # For splitting data into training and test sets 
import joblib  # For saving and loading models
from joblib import Parallel, delayed  # For parallel processing
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as stop_words  # For using predefined English stop words
from sklearn.feature_extraction import text  # For text processing utilities in sklearn
import pickle # For serializing and de-serializing Python objects (saving/loading)


# Configuring the logging format and level
logging.basicConfig(format='%(asctime)s  %(message)s', 
    datefmt='%y-%m-%d %H:%M:%S', level=logging.INFO)

# Defining English stop words 
stop_words = text.ENGLISH_STOP_WORDS

# List of US states
us_states =  ['alabama', 'alaska','arizona', 'arkansas', 'california', 'colorado', 'connecticut', 'delaware', 'florida', 'georgia', 'hawaii', 'idaho', 'illinoi', 'indiana', 'iowa', 
              'kansas','kentucky','louisiana', 'maine', 'maryland', 'massachusetts', 'michigan', 'minnesota', 'mississippi', 'missouri', 'montana',  'nebraska', 'nevada', 
              'new_hampshire', 'new_jersey', 'new_mexico', 'new_york', 'north_carolina', 'north_dakota', 'ohio', 'oklahoma', 'oregon', 'pennsylvania', 'rhode_island', 
              'south_carolina', 'south_dakota', 'tennessee', 'texas', 'utah', 'vermont', 'virgnia', 'washington', 'west_virginia', 'wisconsin', 'wyoming']

# List of US cities
us_cities = ['montgomery', 'juneau', 'phoenix', 'little_rock', 'sacramento', 'denver', 'harford', 'dover', 'tallahassee', 'atlanta', 'boise', 
             'springfield', 'indianapolis', 'des_moines', 'topeka', 'frankfort', 'baton_rouge', 'augusta', 'annapolis', 'boston', 'lansing', 
             'saint_paul', 'jackson', 'jefferson_city', 'helena', 'lincoln', 'carson_city', 'concord', 'trenton', 'santa_fe', 'albany', 'raleigh', 
             'bismarck', 'columbus', 'oklahoma_city', 'salem', 'harrisburg', 'providence', 'columbia', 'pirre', 'nashville', 'austin', 'salt_lake_city', 
             'montpelier', 'richmond', 'olympia', 'charleston', 'madison', 'cheyenne']

# Directory setup
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Gets the 'tax-classification' directory
data_dir = os.path.join(base_dir, 'data')  # Path to the 'data' directory
model_input_dir = os.path.join(base_dir, 'model_input')  # Path to the 'model_input' directory

# Loading a pickle file containing a mapping from phrases to integers
phrase2int_path = os.path.join(model_input_dir, 'statutes-phrase2int.pkl')
phrase2int = pd.read_pickle(phrase2int_path)              
phrase_dict = {value:key for key,value in phrase2int.items()}

# Loading additional data from pickle files
phrase_state_path = os.path.join(data_dir, 'phrase_half.pickle')
data_lexis_path = os.path.join(data_dir, 'df_lexis.pickle')
phrase_state = pickle.load(open(phrase_state_path, 'rb'))
data_lexis = pickle.load(open(data_lexis_path, 'rb'))

# Logging the length of phrase_state
logging.info(str(len(phrase_state)))

# Extracting phrases from data_lexis
phrase_lexis = [i for i in data_lexis.phrases]

# Combining the phrases from data_lexis and phrase_state
phrase_all = phrase_lexis + phrase_state

# List to store the processed phrases
phrase_lst = []

# Processing each phrase
for i in phrase_all:
    w = ' '
    for j in i:
        w += phrase_dict[j] + ' '
    phrase_lst.append(w)

# Initializing a TF-IDF vectorizer with specified parameters
tfidf = TfidfVectorizer(max_features=50000, max_df=0.75, stop_words=list(stop_words) + us_states + us_cities)

# Fitting the vectorizer to the phrase list and transforming it
X = tfidf.fit_transform(phrase_lst)

# Saving the fitted vectorizer for later use
pickle.dump(tfidf, open("model_input/tfidf_both_50000.pickle", "wb"))

