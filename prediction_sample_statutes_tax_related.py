# Importing necessary libraries
import pandas as pd  # Data manipulation and analysis
import re  # Regular expression operations
import nltk  # Natural language processing toolkit
import psycopg2  # PostgreSQL database adapter for Python
import sys  # System-specific parameters and functions
import pickle  # For serializing and de-serializing Python object structures
from nltk.tag.perceptron import PerceptronTagger  # Perceptron tagger for part-of-speech tagging
import numpy as np  # Numerical operations
from download import download_file_from_google_drive  # Function to download files from Google Drive
import os  # Miscellaneous operating system interfaces

# Initialize a Perceptron tagger for part-of-speech tagging
tagger = PerceptronTagger()

# Tokenization of words
from nltk.tokenize import word_tokenize

# Custom module for phrase processing
from pos_phrases import PosPhrases
phraser = PosPhrases()

# Load vocabulary for the phraser from a pickle file
voc = pd.read_pickle('model_input/phraserloglog-vocab.pkl')
phraser.vocab = voc

# Load a pre-trained TF-IDF vectorizer from a pickle file
tfidf = pickle.load(open('model_input/tfidf_both_50000.pickle', 'rb'))

# Initialize empty lists to store sentences, phrases, states, and years
sentences_all = []
phrases_all = []
state_all = []
year_all = []

# Load Chi-Square feature selector and the best logistic regression model for predictions
chi_square_selector = pickle.load(open('tax_related/Logistic/chi2_selector_tfidf_2000.pickle', 'rb'))
best_model = pickle.load(open('tax_related/Logistic/best_model_calibrated_2000.pickle', 'rb'))

# Load the dataset from a pickle file
df = pickle.load(open('data/df_sample1000.pickle', 'rb'))

# Processing each row in the DataFrame
for i in range(0, len(df)):
    statutes = df.iloc[i]['statutes']
    for sent in statutes:
        if len(sent) > 10:
            # Extract words using regular expression
            s = re.findall(r"[a-z]+-?[a-z]+", sent.lower(), flags=re.UNICODE)
            # Remove consecutive duplicate words
            words = [w for i, w in enumerate(s) if s[i] != s[i-1]]
            # Generate phrases from the list of words
            phrases = phraser.phrase(words, ignore_POS=True)
            w = ' '
            if phrases is not None and len(phrases) > 10:
                for j in phrases:
                    w += j + ' '
                # Append sentences and their corresponding phrases to lists
                sentences_all.append(sent)
                phrases_all.append(w)

# Transform phrases with TF-IDF vectorizer and select features with Chi-Square
tfidf_X = tfidf.transform(phrases_all)
X = chi_square_selector.transform(tfidf_X)

# Predict probabilities using the best model
prediction_all = best_model.predict_proba(X)[:,1]

# Creating a new DataFrame to store results
df_all = pd.DataFrame()  # Initialize an empty DataFrame
df_all['Sentences'] = sentences_all  # Add sentences to the DataFrame
df_all['Phrases'] = phrases_all  # Add processed phrases to the DataFrame
df_all['prediction'] = prediction_all  # Add prediction probabilities to the DataFrame

# Check if the directory 'Prediction' exists, if not, create it
if not os.path.exists('Prediction'):
    os.makedirs('Prediction')  # Creates the 'Prediction' directory

# Save the DataFrame to a CSV file in the 'Prediction' directory
df_all.to_csv('Prediction/df_sentences.csv')  # Save df_all as a CSV file
