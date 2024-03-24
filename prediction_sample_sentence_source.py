# Importing necessary libraries
import pandas as pd  # Data manipulation and analysis library
import re  # Library for regular expression operations
import nltk  # Natural Language Toolkit for text processing and analysis
import sys  # System-specific parameters and functions
import pickle  # Module to serialize and deserialize Python object structures
from nltk.tag.perceptron import PerceptronTagger  # Perceptron-based part-of-speech tagger
import numpy as np  # Library for numerical operations
from download import download_file_from_google_drive  # Custom module to download files from Google Drive
from nltk.tokenize import word_tokenize  # Function to split text into tokens (words)
import os  # Module for interacting with the operating system
from pos_phrases import PosPhrases  # Custom module for phrase processing

# Custom module for phrase processing
phraser = PosPhrases()

# Load vocabulary for the phraser from a pickle file
voc = pd.read_pickle('model_input/phraserloglog-vocab.pkl')
phraser.vocab = voc

# Check if the directory 'Prediction' exists, if not, create it
if not os.path.exists('Prediction'):
    os.makedirs('Prediction')  # Creates the 'Prediction' directory

#prediction_path = 'Prediction'
#os.chdir(prediction_path)

# Load a pre-trained TF-IDF vectorizer from a pickle file
tfidf = pickle.load(open('model_input/tfidf_both_50000.pickle', 'rb'))

# Load Chi-Square feature selector and the best random forest model for predictions
chi_square_selector = pickle.load(open('tax_source/chi2_selector_tfidf_30000.pickle','rb'))
best_model = pickle.load(open('best_model_tfidf_all_calibrator_30000.pickle','rb'))

# Loading a pickle file containing a mapping from phrases to integers
phrase2int = pd.read_pickle('model_input/statutes-phrase2int.pkl')         
phrase_dict = {value:key for key,value in phrase2int.items()}

# Initialize empty lists to store sentences, phrases and tax sources 
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

# Load the dataset from a pickle file
df = pickle.load(open('data/df_sample1000.pickle','rb'))

# Processing each row in the DataFrame
for i in range(0,len(df)):
    statutes = df.iloc[i]['statutes']
    for sent in statutes:
        if len(sent)>10:
            # Extract words using regular expression
            s = re.findall(r"[a-z]+-?[a-z]+", sent.lower(),flags = re.UNICODE)
            # Remove consecutive duplicate words
            words = [w for i,w in enumerate(s) if s[i] != s[i-1]]
            # Generate phrases from the list of words
            phrases = phraser.phrase(words, ignore_POS = True) 
            w=' '
            if phrases is not None:
                if len(phrases)>10:
                    for j in phrases:
                        w+=j+' '
                    # Append sentences and their corresponding phrases to lists    
                    sentences_all.append(sent)
                    phrases_all.append(w)

# Transform phrases with TF-IDF vectorizer and select features with Chi-Square
tfidf_X = tfidf.transform(phrases_all)  
X = chi_square_selector.transform(tfidf_X)

# Predict probabilities using the best model
prediction_all = best_model.predict(X)

# Store predicted probabilities for each class
for i in list(class_mapping.keys()):
  class_mapping[i] = class_mapping[i] +list(best_model.predict_proba(X)[:,i]) 

# Creating a new DataFrame to store results
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

# Check if the directory 'Prediction' exists, if not, create it
if not os.path.exists('Prediction'):
    os.makedirs('Prediction')  # Creates the 'Prediction' directory
  
# Save the DataFrame to a CSV file in the 'Prediction' directory
df_all.to_csv('Prediction/df_sentences_source.csv')
