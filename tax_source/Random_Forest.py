# Import necessary libraries
import pandas as pd  # Import pandas for data manipulation
import numpy as np   # Import numpy for numerical operations
import itertools     # Import itertools for efficient looping and combination generation
import os            # Import os for operating system related functionalities
import re            # Import re for regular expressions
import logging       # Import logging for logging messages during program execution
import gc            # Import gc for garbage collection

from sklearn.feature_extraction.text import TfidfVectorizer  # Import TfidfVectorizer for text feature extraction
from sklearn.model_selection import train_test_split         # Import train_test_split for splitting data into train and test sets
from ast import literal_eval                                 # Import literal_eval for safely evaluating strings containing Python expressions
from sklearn.ensemble import RandomForestClassifier          # Import RandomForestClassifier for building Random Forest models
from sklearn.ensemble import ExtraTreesClassifier            # Import ExtraTreesClassifier for building Extra Trees models
from sklearn.model_selection import cross_val_score          # Import cross_val_score for cross-validation evaluation
from sklearn.calibration import calibration_curve            # Import calibration_curve for generating calibration curves
import matplotlib                                            # Import matplotlib for data visualization
matplotlib.use('Agg')                                        # Set the backend of matplotlib for non-interactive use
import matplotlib.pyplot as plt                              # Import pyplot module for plotting
from matplotlib.offsetbox import AnchoredText                # Import AnchoredText for annotation in plots
import matplotlib.lines as mlines                            # Import mlines for adding lines to plots
import matplotlib.transforms as mtransforms                  # Import mtransforms for transforming coordinates in plots
from sklearn.metrics import confusion_matrix                 # Import confusion_matrix for computing confusion matrix
from sklearn.metrics import multilabel_confusion_matrix      # Import multilabel_confusion_matrix for computing multi-label confusion matrix
import pickle                                                # Import pickle for serializing and deserializing Python objects
from sklearn.calibration import CalibratedClassifierCV       # Import CalibratedClassifierCV for probability calibration
from collections import Counter                              # Import Counter for counting occurrences of elements in a list
from sklearn.feature_extraction import DictVectorizer        # Import DictVectorizer for converting dictionaries into feature matrices
from sklearn.feature_extraction.text import TfidfTransformer  # Import TfidfTransformer for transforming count matrices to TF-IDF representation
from sklearn.feature_selection import SelectKBest            # Import SelectKBest for selecting features
from sklearn.feature_selection import chi2                   # Import chi2 for chi-squared statistic computation
from sklearn.model_selection import GridSearchCV             # Import GridSearchCV for hyperparameter tuning
from sklearn.metrics import accuracy_score                   # Import accuracy_score for computing accuracy
from sklearn.metrics import f1_score                         # Import f1_score for computing F1 score
from sklearn import metrics                                  # Import metrics for evaluation metrics computation
from sklearn.metrics import roc_auc_score                    # Import roc_auc_score for computing ROC AUC score
from sklearn.metrics import precision_score                  # Import precision_score for computing precision
from sklearn.metrics import recall_score                     # Import recall_score for computing recall
from sklearn.metrics import classification_report            # Import classification_report for generating a classification report
from sklearn.feature_extraction import text                  # Import text for text feature extraction utilities
from imblearn.over_sampling import SMOTE                     # Import SMOTE for oversampling imbalanced data


## Set logging format
logging.basicConfig(format='%(asctime)s  %(message)s', 
    datefmt='%y-%m-%d %H:%M:%S', level=logging.INFO)

# Function to create calibration plot
def calibration_plot(x,y,name,prediction_path,calibrated):

  """
  Create a calibration plot.
  Args:
  x (array): Array of x values.
  y (array): Array of y values.
  name (str): Name of the plot.
  prediction_path (str): Path to save the plot.
  calibrated (bool): Whether the model is calibrated.

  Returns:
  None  

  """
  fig, ax = plt.subplots()
  # only these two lines are calibration curves
  plt.plot(x,y, marker='o', linewidth=1, label='Random Forest')  
  
  # reference line, legends, and axis labels
  line = mlines.Line2D([0, 1], [0, 1], color='black')
  transform = ax.transAxes
  line.set_transform(transform)
  ax.add_line(line)
  fig.suptitle('Calibration plot for '+name)
  ax.set_xlabel('Predicted probability')
  ax.set_ylabel('True probability in each bin')
  plt.legend()

  # Construct the file path
  filename = f'Calibration_Plot_Calibrated_{name}.pdf' if calibrated else f'Calibration_Plot_{name}.pdf'
  full_file_path = os.path.join(prediction_path, filename)

  # Save the figure
  fig.savefig(full_file_path)
  print(f"File saved at: {full_file_path}")

  #print("Current Directory Before Changing:", os.getcwd())  # Print the current working directory
  #os.chdir(prediction_path)
  #print("Directory After Changing:", os.getcwd())  # Print the directory after changing it
  #if calibrated == True:
  #  fig.savefig(prediction_path+'/Calibration_Plot_Calibrated_'+str(name)+'.pdf')
  #else:
  #  fig.savefig(prediction_path+'/Calibration_Plot_'+str(name)+'.pdf')
    

# Function to create features    
def create_features(data,k,phrase_dict,prediction_path):  

  """
  Create features.

  Args:
  data (DataFrame): Input data.
  k (int): Number of features to select.
  phrase_dict (dict): Dictionary mapping phrase indices to phrases.
  prediction_path (str): Path to save the features.

  Returns:
  X (array): Feature matrix.
  y (array): Target labels.
  features_selected (array): Selected feature names.

  """
  phrase= data['phrase'].values #[i[2] for i in data]
  dc_identifiers = data['dc_identifiers'].values #[i[0] for i in data]
  y = data['label_source'].values#[i[1] for i in data] # label
  phrase_lst = []

  # Convert phrase indices to phrases
  for i in phrase:
    w=' '
    for j in i:
      w+= phrase_dict[j]+' '
    phrase_lst.append(w)

  # Load TF-IDF model and transform phrases into feature vectors
  tfidf = pickle.load(open('path_to_your_tax_classification_folder/model_input/tfidf_both_50000.pickle','rb'))
  X = tfidf.fit_transform(phrase_lst)

  del phrase,data
  gc.collect()

  logging.info("Features created")

## Select features using Chi-squared
  chi2_selector = SelectKBest(chi2, k=k)
  X = chi2_selector.fit_transform(X, y)

  ## Get selected feature names
  support=chi2_selector.get_support()
  feature_names = tfidf.get_feature_names()
  #v.restrict(support)
  features_selected = np.array(feature_names)[support]
  
  pickle.dump(tfidf, open(prediction_path+"/tfidf_vec"+str(k)+".pickle", "wb"))
  pickle.dump(chi2_selector, open(prediction_path+"/chi2_selector_tfidf"+str(k)+".pickle", "wb"))

  return X,y,features_selected

# Function to train and save the model
def model_train_save(X,y, features_selected,prediction_path,phrase_dict,k, calibrated,upsampling=False,SMOTE_Sampling=False,ET=False):
  """ 
  Train and save the model.

  Args:
  X (array): Feature matrix.
  y (array): Target labels.
  features_selected (array): Selected feature names.
  prediction_path (str): Path to save the model.
  phrase_dict (dict): Dictionary mapping phrase indices to phrases.
  k (int): Number of features selected.
  calibrated (bool): Whether the model is calibrated.
  upsampling (bool): Whether to perform upsampling.
  SMOTE_Sampling (bool): Whether to perform SMOTE sampling.
  ET (bool): Whether to use Extra Trees classifier.

  Returns:
  None
  """
  # Split data into training and testing sets
  train_sample_size = int(0.8*len(y))
  indices = list(range(len(y)))
  X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X, y, indices, train_size = train_sample_size, random_state=0,stratify=y) 

  # Perform upsampling if specified
  if upsampling == True:
    y_train=np.array(y_train)
    ind_0 = np.where(y_train==0)[0]
    ind_1 = np.where(y_train==1)[0]
    num_class0=len(ind_0)
    num_class1=len(ind_1)
    ind_class1_upsampled = np.random.choice(ind_1, size=num_class0, replace=True)
    ind_new = np.concatenate((ind_class1_upsampled, ind_0))
    y_train = y_train[ind_new]
    X_train = X_train[ind_new]

  # Perform SMOTE sampling if specified
  if SMOTE_Sampling == True:
    smt = SMOTE(sampling_strategy='auto')
    X_train, y_train = smt.fit_sample(X_train, y_train)

  logging.info("train, test sets created")  
                                                                                     
  # Build a Random Forest or Extra Trees model  
  RF = RandomForestClassifier(oob_score= True,warm_start = False, class_weight = "balanced", random_state=0)
  ET_Classifier = ExtraTreesClassifier(random_state = 0)

  # Set parameters for Grid Search  
  n_estimators = [500,800]
  max_depth = [50,35]
  max_features=[20]
  criterion=['entropy']
  hyperparameters = dict(criterion=criterion, n_estimators = n_estimators, max_depth = max_depth,max_features=max_features)
  
  # Choose between Random Forest and Extra Trees
  if ET == True:
    ensemble = ET_Classifier
    print("Extra Tree")
  else:
    ensemble = RF
    print("Random Forest")

  # Perform Grid Search for hyperparameter tuning  
  if calibrated == False:
    clf = GridSearchCV(ensemble, hyperparameters, cv=5, verbose=2, n_jobs=-1, pre_dispatch = 32)
    
    logging.info("Grid Search Done")
    
    clf.fit(X_train, y_train)
    best_model = clf.best_estimator_
  
  # Calibrate the model if specified  
  if calibrated == True:     
    if ET == False:
      model = RandomForestClassifier(random_state = 0, max_features=50, n_estimators=500, max_depth=100, class_weight = "balanced" )
    else:
      model = ExtraTreesClassifier(random_state = 0, max_features=50, n_estimators=500, max_depth=100, class_weight = "balanced" )
      
    calibrated = CalibratedClassifierCV(model, method='sigmoid', cv=5)
    calibrated.fit(X_train,y_train)
    best_model = calibrated
    
  
  # Print best parameters if model is not calibrated
  if calibrated == False:
    print('Criterion: ' + str(best_model.get_params()['criterion']))
    print('max_depth: ' + str(best_model.get_params()['max_depth']))
    print('n_estimator: ' + str(best_model.get_params()['n_estimators']))
    print('max_features: '  + str(best_model.get_params()['max_features']))


  # Make predictions and calculate probabilities    
  y_test_predict = best_model.predict(X_test)
  y_train_predict = best_model.predict(X_train)
  y_prob = best_model.predict_proba(X_test)
  y_prob_train = best_model.predict_proba(X_train)

  # Generate calibration plot for each class  
  for j in range(0,len(np.unique(y))):
    rf_y, rf_x = calibration_curve([1 if i==np.unique(y)[j] else 0 for i in y_test], y_prob[:,j], n_bins=10)
    calibration_plot(rf_x,rf_y,'Random_Forest_'+np.unique(y)[j],prediction_path,calibrated=True)  
   
  # Print evaluation metrics
  print(y_prob_train[0])  # Print the predicted probabilities of the first sample in the training set
  train_acc = accuracy_score(y_train, y_train_predict)  # Calculate and print the training accuracy
  test_acc = accuracy_score(y_test, y_test_predict)  # Calculate and print the testing accuracy
  print("Train Accuracy: " + str(train_acc))
  print("Test Accuracy: " + str(test_acc))

  f1_train = f1_score(y_train, y_train_predict, labels=y_train, average='macro')  # Calculate and print the training F1 score
  f1_test = f1_score(y_test, y_test_predict, labels=y_test, average='macro')  # Calculate and print the testing F1 score
  print("Train F1: " + str(f1_train))
  print("Test F1: " + str(f1_test))
  
  precision_train = precision_score(y_train, y_train_predict, labels=y_train, average='macro')  # Calculate and print the training precision
  precision_test = precision_score(y_test, y_test_predict, labels=y_test, average='macro')  # Calculate and print the testing precision
  print("Train precision: " + str(precision_train))
  print("Test precision: " + str(precision_test))
  
  recall_train = recall_score(y_train, y_train_predict, labels=y_train, average='macro')  # Calculate and print the training recall
  recall_test = recall_score(y_test, y_test_predict, labels=y_test, average='macro')  # Calculate and print the testing recall
  print("Train recall: " + str(recall_train))
  print("Test recall: " + str(recall_test))
  
  confusion_matrix_result = confusion_matrix(y_test, y_test_predict, labels=list(np.unique(y)))  # Calculate and print the confusion matrix
  confusion_matrix_multi = multilabel_confusion_matrix(y_test, y_test_predict, labels=list(np.unique(y)))  # Calculate and print the multilabel confusion matrix
  print(confusion_matrix_result)
  print(confusion_matrix_multi)
  
  if calibrated == False:
    importance = best_model.feature_importances_  # Extract feature importances if the model is not calibrated
    ranking = np.argsort(importance)[::-1]
    features_sorted = np.array(features_selected)[ranking]
    features = [phrase_dict[i] for i in features_sorted]
  
    print("Top 20 features: ", features[0:20])  # Print the top 20 features
    print("Top 20 features magnitude percentage", importance[0:20] / np.sum(importance))  # Print the percentage contribution of top 20 features
  
  logging.info("Results printed")

  # Dump the best model and grid search object if not calibrated
  pickle.dump(best_model, open(prediction_path + "/best_model_tfidf" + str(k) + ".pickle", "wb"))
  if calibrated == False:
    pickle.dump(clf, open(prediction_path + "/clf_tfidf" + str(k) + ".pickle", "wb"))
  
  logging.info("model saved")
  
  # Define the file name for saving results based on sampling techniques
  if upsampling == True:
    file_name = prediction_path + '/result_RF_' + str(k) + '_upsampled' + '.txt'
    if  ET == True:
      file_name = prediction_path + '/result_ET_' + str(k) + '.txt'
  elif SMOTE_Sampling == True:
    file_name = prediction_path + '/result_RF_' + str(k) + '_SMOTE' + '.txt'
  else:
    file_name = prediction_path + '/result_RF_' + str(k) + '.txt'
  
  # Write evaluation results to a text file
  with open(file_name, "w") as text_file:
    print("Total datapoints: " + str(len(y)) + "\n", file=text_file)
    print("Training set: " + str(len(y_train)) + "\n", file=text_file)
    print("Train Accuracy: " + str(train_acc), file=text_file)
    print("Test Accuracy: " + str(test_acc), file=text_file)
    print("Train F1: " + str(f1_train), file=text_file)
    print("Test F1: " + str(f1_test), file=text_file)
    print("Train precision: " + str(precision_train), file=text_file)
    print("Test precision: " + str(precision_test), file=text_file)
    print("Train recall: " + str(recall_train), file=text_file)
    print("Test recall: " + str(recall_test), file=text_file)
    print(confusion_matrix_result, file=text_file)
    print(confusion_matrix_multi, file=text_file)
    print(classification_report(y_test, y_test_predict), file=text_file)
    if calibrated == False:
      print(features[0:20], file=text_file)  # Write the top 20 features to the file
      print(importance[0:20] / np.sum(importance), file=text_file)  # Write the feature importance percentages to the file  
  

def main():
  ## Load data from saved pickle files
  # Changing the working directory to two levels up from the current directory
  #os.chdir("../..")
  # Printing the current working directory for verification
  print("Current Working Directory:", os.getcwd())

  # Setting the path for tax source data and the path for storing predictions
  path = "tax_source"
  prediction_path = "Prediction"
  # Creating the prediction path directory if it does not exist
  os.makedirs(prediction_path, exist_ok=True)  

  # Read the data from CSV file  
  data = pd.read_csv('path_to_your_tax_classification_folder/tax_source/df_final_new.csv',index_col=0)
  data = data.replace(np.nan, 'Other', regex=True)

  # Convert the 'phrase' column from string to list of phrases
  data.phrase=data['phrase'].apply(literal_eval)  
  
  logging.info("Data loaded")
   
  # Load phrase to integer mapping    
  phrase2int = pd.read_pickle('path_to_your_tax_classification_folder/model_input/statutes-phrase2int.pkl')              
  phrase_dict = {value:key for key,value in phrase2int.items()}
  stop_words = text.ENGLISH_STOP_WORDS

## Remove phrases still in stop_words list  
  for i in range(0,data.shape[0]):
    for j in data['phrase'].iloc[i]:
        if phrase_dict[j] in stop_words:
          data['phrase'].values[i] = list(filter((j).__ne__, data['phrase'].iloc[i]))
          
  k=10000
    
  # Extract features  
  features_return = create_features(data=data,k=k,phrase_dict = phrase_dict,prediction_path=prediction_path)          
  X=features_return[0]
  y=features_return[1]
  features_selected = features_return[2]

  # Train and save the model
  model_train_save(X=X,y=y,features_selected=features_selected,prediction_path=prediction_path,phrase_dict=phrase_dict,k=k,calibrated = True)

# Entry point of the script
if __name__ == "__main__":
    main()
