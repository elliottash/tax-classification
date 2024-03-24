# Import necessary libraries for data manipulation, feature extraction, and machine learning
import pandas as pd                                                        # For data manipulation and analysis
import numpy as np                                                         # For numerical calculations and working with arrays
import itertools                                                           # For efficient looping and iterator algebra
import os                                                                  # For interacting with the operating system, like navigating directories
import re                                                                  # For regular expression matching, useful in text data
import logging                                                             # For logging messages and debugging
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer  # For converting text to vector form based on term frequency-inverse document frequency and hashing techniques
from sklearn.model_selection import train_test_split                       # For splitting dataset into training and testing sets
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier  # For Random Forest and Extra Trees classifier algorithms
from sklearn.model_selection import cross_val_score                        # For evaluating model performance using cross-validation
from sklearn.calibration import calibration_curve                          # For generating calibration curves, comparing predictions to outcomes
import matplotlib                                                          # For creating static, interactive, and animated visualizations in Python
matplotlib.use('Agg')                                                      # Configures matplotlib to use a non-interactive backend, allowing plots to be saved to files
import matplotlib.pyplot as plt                                            # For creating plots and charts
from matplotlib.offsetbox import AnchoredText                              # For adding text boxes to plots
import matplotlib.lines as mlines                                          # For creating line elements in plots
import matplotlib.transforms as mtransforms                                # For low-level transformation utilities
from sklearn.metrics import confusion_matrix                               # For evaluating classification accuracy
import pickle                                                              # For serializing and deserializing Python object structures
from collections import Counter                                            # For counting hashable objects in an efficient way
from sklearn.feature_extraction import DictVectorizer, text                # For converting dictionaries into vector form and working with text data
from sklearn.feature_selection import SelectKBest, chi2                    # For feature selection using statistical tests
from sklearn.model_selection import GridSearchCV                           # For exhaustive search over specified parameter values for an estimator
from sklearn.metrics include accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, classification_report  # For evaluating model performance metrics
from sklearn.calibration import CalibratedClassifierCV                     # For probability calibration of classifiers
from imblearn.over_sampling import SMOTE                                   # For Synthetic Minority Over-sampling Technique to address class imbalance
from sklearn.preprocessing include StandardScaler, Normalizer              # For standardizing and normalizing datasets
import gc                                                                  # For manual garbage collection to manage memory during large computations

# Configuring the logging format for debugging and tracking
logging.basicConfig(format='%(asctime)s  %(message)s', 
    datefmt='%y-%m-%d %H:%M:%S', level=logging.INFO)

# Function to generate calibration plots comparing model predictions to actual outcomes
def calibration_plot(x,y,predict_prob,name,prediction_path,calibrated):
  fig, ax = plt.subplots()
  plt.plot(x,y, marker='o', linewidth=1, label='RF') # Plotting the calibration curve
  plt.hist(predict_prob,weights=np.ones(len(predict_prob)) / len(predict_prob), color = 'orange') # Histogram of predicted probabilities
  line = mlines.Line2D([0, 1], [0, 1], color='black') # Reference line for perfect calibration
  transform = ax.transAxes
  line.set_transform(transform)
  ax.add_line(line)
  fig.suptitle('Calibration plot for '+name)
  ax.set_xlabel('Predicted probability')
  ax.set_ylabel('True probability in each bin')
  plt.legend()
  os.chdir(prediction_path) # Changing directory to save the plot
  # Saving the plot based on whether model is calibrated or not
  if calibrated == True:
    fig.savefig('Calibration_Plot_Calibrated.pdf')
  else:
    fig.savefig('Calibration_Plot.pdf')

# Function to create features from the raw data using TF-IDF or Hashing Vectorizer and perform feature selection using Chi-Squared test
def create_features(data,k,prediction_path,both, phrase_dict, standardize):
  phrase= data.phrases
  dc_identifiers = data.dc_identifier
  y = data.tax_code # Target variable
  phrase_lst = []

  # Transforming phrases using the phrase dictionary
  for i in phrase:
    w=' '
    for j in i:
      w+=phrase_dict[j]+' '
    phrase_lst.append(w)

  # Selecting the vectorization method based on the 'both' parameter
  if both == 1:
    tfidf = pickle.load(open('path_to_your_tax_classification_folder/model_input/tfidf_both_50000.pickle','rb'))
  elif both == 0:
    tfidf = TfidfVectorizer(max_features=20000,max_df=0.75)
  elif both==-1:
    hashing = HashingVectorizer(n_features=2**16,alternate_sign=False)
  
  # Applying the chosen vectorization method
  if both != -1:
    X = tfidf.fit_transform(phrase_lst)
  else:
    X = hashing.fit_transform(phrase_lst)

  # Clearing memory
  del phrase,data
  gc.collect()

  logging.info("Features created")

  # Feature selection using Chi-squared test
  chi2_selector = SelectKBest(chi2, k=k)
  X = chi2_selector.fit_transform(X, y)

  # Optionally standardizing the features
  if standardize == True:
    scaler = Normalizer()
    X = scaler.fit_transform(X)

  # Extracting feature names selected by the Chi-squared test
  if both !=-2:
    support=chi2_selector.get_support()
    feature_names = tfidf.get_feature_names()
    features_selected = np.array(feature_names)[support]
    features_selected = [i for i in features_selected]
    
    # Saving the vectorizer, selector, and selected features for future use
    pickle.dump(tfidf, open(prediction_path+"/tfidf_vec"+str(k)+".pickle", "wb"))
    pickle.dump(chi2_selector, open(prediction_path+"/chi2_selector_tfidf"+str(k)+".pickle", "wb"))
    pickle.dump(features_selected, open(prediction_path+"/feature_selected_"+str(k)+".pickle", "wb"))
  else:
    features_selected = []
    pickle.dump(hashing, open(prediction_path+"/hashing_vec"+str(k)+".pickle", "wb"))
    pickle.dump(chi2_selector, open(prediction_path+"/chi2_selector_tfidf"+str(k)+".pickle", "wb"))
  return X,y,features_selected

# Function to train the model, perform hyperparameter tuning, evaluate performance, and save the model and results
def model_train_save(X,y, features_selected,prediction_path,phrase_dict,k,calibrated,ET,upsampling=False,SMOTE_Sampling=False, standardize=True):
  # Setting the prediction path based on model type
  if ET == True:
    prediction_path = "tax_related/RF/Prediction/ET"
  
  # Splitting the data into training and test sets
  train_sample_size = int(0.8*len(y))
  indices = list(range(len(y)))
  X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X, y, indices, train_size = train_sample_size, random_state=0,stratify=y) 

  # Handling class imbalance by upsampling or using SMOTE
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

  if SMOTE_Sampling == True:
    smt = SMOTE(sampling_strategy='auto')
    X_train, y_train = smt.fit_sample(X_train, y_train)

  logging.info("train, test sets created")

  # Training the model
  print("Calibrated: " + str(calibrated))
  if calibrated == False:
    # Configure Random Forest Classifier with balanced class weights for handling imbalance in data
    RF = RandomForestClassifier(oob_score= True, warm_start = False, class_weight = "balanced", random_state=0)
    ET_Classifier = ExtraTreesClassifier(random_state = 0)

    # Parameters for GridSearchCV
    n_estimators = [500, 800, 1000]
    max_depth = [35, 50]
    max_features = [35, 50]
    criterion = ['entropy']
    hyperparameters = dict(criterion=criterion, n_estimators=n_estimators, max_depth=max_depth, max_features=max_features)

    # Choose the classifier based on ET flag
    if ET == True:
      ensemble = ET_Classifier
      print("Extra Tree")
    else:
      ensemble = RF
      print("Random Forest")

    # Perform grid search to find the best hyperparameters  
    clf = GridSearchCV(ensemble, hyperparameters, cv=5, verbose=2, n_jobs=-1, pre_dispatch = 32)
    
    logging.info("Grid Search Done")
    
    clf.fit(X_train, y_train)
    best_model = clf.best_estimator_

  # Calibrate model if required
  if calibrated == True:
    X_train_new, X_val, y_train_new, y_val = train_test_split(X_train, y_train, train_size = int(0.8*train_sample_size), random_state=0,stratify=y_train) 
    if ET == False:
      model = RandomForestClassifier(oob_score= True,warm_start = False, class_weight = "balanced", random_state=0,n_estimators=800, max_depth = 50, max_features=35,criterion='entropy')
 
    else:
      model = ExtraTreesClassifier(random_state = 0, n_estimators=800, max_depth = 50, max_features = 35,criterion = 'entropy', n_jobs = -1)
      
    #model.fit(X_train,y_train)
    calibrated = CalibratedClassifierCV(model, method='sigmoid', cv=5)
    calibrated.fit(X_train,y_train)
    best_model = calibrated  

  # Output the best model parameters
  if not calibrated:
    print(f'Criterion: {best_model.get_params()["criterion"]}')
    print(f'max_depth: {best_model.get_params()["max_depth"]}')
    print(f'n_estimator: {best_model.get_params()["n_estimators"]}')
    print(f'max_features: {best_model.get_params()["max_features"]}')

  # Make predictions on the test set
  y_test_predict = best_model.predict(X_test)
  y_train_predict = best_model.predict(X_train)
  y_prob = best_model.predict_proba(X_test)
  y_prob_train = best_model.predict_proba(X_train)

  # Generate calibration curve and plot
  rf_y, rf_x = calibration_curve(y_test, y_prob[:,1], n_bins=20, strategy='uniform')
  calibration_plot(rf_x, rf_y, y_prob[:,1], 'Random Forest', prediction_path, calibrated=True)
  print('Calibration plot created')

  # Calculate and print performance metrics
  train_acc = accuracy_score(y_train, y_train_predict)
  test_acc = accuracy_score(y_test, y_test_predict)
  print(f"Train Accuracy: {train_acc}")
  print(f"Test Accuracy: {test_acc}")
  
  # More metrics
  f1_train = f1_score(y_train, y_train_predict, labels=np.unique(y_train))
  f1_test = f1_score(y_test, y_test_predict, labels=np.unique(y_test))
  print(f"Train F1: {f1_train}")
  print(f"Test F1: {f1_test}")

  # Precision, Recall, and AUCROC
  precision_train = precision_score(y_train, y_train_predict, labels=np.unique(y_train))
  precision_test = precision_score(y_test, y_test_predict, labels=np.unique(y_test))
  print(f"Train precision: {precision_train}")
  print(f"Test precision: {precision_test}")

  recall_train = recall_score(y_train, y_train_predict, labels=np.unique(y_train))
  recall_test = recall_score(y_test, y_test_predict, labels=np.unique(y_test))
  print(f"Train recall: {recall_train}")
  print(f"Test recall: {recall_test}")

  aucroc = roc_auc_score(y_test, y_prob[:,1])
  aucroc_train = roc_auc_score(y_train, y_prob_train[:,1])
  print(f"Train AUCROC: {aucroc_train}")
  print(f"Test AUCROC: {aucroc}")

  # Confusion matrix
  tn, fp, fn, tp = confusion_matrix(y_test, y_test_predict).ravel()
  confusion_matrix_result = f"Confusion matrix\nTrue negatives: {tn}\nFalse positives: {fp}\nFalse negatives: {fn}\nTrue positives: {tp}\n"
  print(confusion_matrix_result)

  # Feature importance for RandomForest and ExtraTrees models
  if not calibrated and len(features_selected) > 1:
    importance = best_model.feature_importances_
    ranking = np.argsort(importance)[::-1]
    features_sorted = np.array(features_selected)[ranking]
    features = [i for i in features_sorted]

    print("Top 50 features: ", features[:50])
    print("Top 50 features magnitude percentage", importance[:50] / np.sum(importance))

  logging.info("Results printed")

  # Determines file name suffix based on whether data was standardized  
  if standardize == True:
    append = 'standardized'
  else:
    append = ''
  
  # Saves the model; path differs based on calibration status
  if calibrated == False:
    pickle.dump(clf,open(prediction_path+"/best_model_no_calibration_"+str(k)+append+".pickle","wb"))
  else:
    pickle.dump(best_model,open(prediction_path+"/best_model_tfidf_all_calibrator_"+str(k)+append+".pickle","wb"))  
  
  logging.info("model saved")  # Logs that the model has been saved
  
  # Determines the file name for results, taking into account upsampling, SMOTE, and whether the model is an Extra Trees classifier
  if upsampling == True:
    file_name = prediction_path+'/result_RF_all_calibrator'+str(k)+'_upsampled'+append+'.txt'
    if  ET == True:
      file_name = prediction_path +'/result_ET_all_calibrator'+str(k)+append+'.txt'
  elif SMOTE_Sampling == True:
    file_name = prediction_path+'/result_RF_all_calibrator'+str(k)+'_SMOTE'+append+'.txt'

  else:
    file_name = prediction_path+'/result_RF_all_calibrator'+str(k)+append+'.txt'
  
  # Opens the results file for writing and logs various performance metrics
  with open(file_name, "w") as text_file:
    print("Total datapoints: "+str(len(y))+"\n",file=text_file)
    print("Training set: "+str(len(y_train))+"\n",file=text_file)
    print("Train Accuracy: " + str(train_acc),file=text_file)
    print("Test Accuracy: " + str(test_acc),file=text_file)
    print("Train F1: " + str(f1_train),file=text_file)
    print("Test F1: " + str(f1_test),file=text_file)
    print("Train precision: " + str(precision_train),file=text_file)
    print("Test precision: " + str(precision_test),file=text_file)
    print("Train recall: " + str(recall_train),file=text_file)
    print("Test recall: " + str(recall_test),file=text_file)
    print("Train AUCROC: "+str(aucroc_train),file=text_file)
    print("Test AUCROC: " + str(aucroc),file=text_file)
    print(confusion_matrix_result,file=text_file)
    print(classification_report(y_test, y_test_predict),file=text_file)
    if calibrated==False and len(features_selected)>1:
      print(features[0:50],file=text_file)
      print(importance[0:50]/np.sum(importance),file=text_file )
    
# Main function to execute the workflow
def main():
  
  # Changing the working directory to two levels up from the current directory
  os.chdir("../..")
  
  # Printing the current working directory for verification
  print("Current Working Directory:", os.getcwd())  # Verifies current working directory
  
  path = "tax_related"
  prediction_path = "path_to_your_tax_classification_folder/tax_related/RF/Prediction"
  os.chdir(path)
  # Creating the prediction path directory if it does not exist
  os.makedirs(prediction_path, exist_ok=True)  

  # Loads data and auxiliary files
  data = pickle.load(open('path_to_your_tax_classification_folder/data/df_lexis.pickle',"rb")) # a list   
  logging.info("Data loaded")    
  phrase2int = pd.read_pickle('path_to_your_tax_classification_folder/model_input/statutes-phrase2int.pkl')              
  phrase_dict = {value:key for key,value in phrase2int.items()}
  stop_words = text.ENGLISH_STOP_WORDS
          
  k=30000    # Feature selection parameter

  # Generates features from the data  
  features_return = create_features(data=data,k=k,prediction_path=prediction_path,both=1, phrase_dict = phrase_dict, standardize = False)
          
  X=features_return[0]
  y=features_return[1]
  features_selected = features_return[2]
  # Trains the model and saves the results
  model_train_save(X=X,y=y,features_selected=features_selected,prediction_path=prediction_path,phrase_dict=phrase_dict,k=k,ET=False,calibrated=True,upsampling=False, standardize= False)

if __name__ == "__main__":
    main()
  

  
