import pandas as pd
import numpy as np
import itertools
import os
import re
import logging 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import train_test_split  

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
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, Normalizer


## Set logging format
logging.basicConfig(format='%(asctime)s  %(message)s', 
    datefmt='%y-%m-%d %H:%M:%S', level=logging.INFO)


def calibration_plot(x,y,predict_prob,name,prediction_path,calibrated):
  fig, ax = plt.subplots()
  # only these two lines are calibration curves
  plt.plot(x,y, marker='o', linewidth=1, label='RF')
  plt.hist(predict_prob,weights=np.ones(len(predict_prob)) / len(predict_prob), color = 'orange')
  # reference line, legends, and axis labels
  line = mlines.Line2D([0, 1], [0, 1], color='black')
  transform = ax.transAxes
  line.set_transform(transform)
  ax.add_line(line)
  fig.suptitle('Calibration plot for '+name)
  ax.set_xlabel('Predicted probability')
  ax.set_ylabel('True probability in each bin')
  plt.legend()
  os.chdir(prediction_path)
  
  if calibrated == True:
    fig.savefig('Calibration_Plot_Calibrated.pdf')
  else:
    fig.savefig('Calibration_Plot.pdf')


def create_features(data,k,prediction_path,both, phrase_dict, standardize):  
  phrase= data.phrases
  dc_identifiers = data.dc_identifier
  y = data.tax_code # label
  phrase_lst = []

  for i in phrase:
    w=' '
    for j in i:
      w+=phrase_dict[j]+' '
    phrase_lst.append(w)

  
  if both == 1:
    tfidf = pickle.load(open('/model_input/tfidf_both_50000.pickle','rb'))
  if both == 0:
    tfidf = TfidfVectorizer(max_features=20000,max_df=0.75)
  if both==-1:
    hashing = HashingVectorizer(n_features=2**16,alternate_sign=False)
    
  if both != -1:
    X = tfidf.fit_transform(phrase_lst)
  else:
    X = hashing.fit_transform(phrase_lst)

  del phrase,data
  gc.collect()

  logging.info("Features created")

  ## Select features using Chi sqaured

  chi2_selector = SelectKBest(chi2, k=k)
  X = chi2_selector.fit_transform(X, y)

  if standardize == True:
    scaler = Normalizer()
    X = scaler.fit_transform(X)

  ## Get selected feature names
  if both !=-2:
    support=chi2_selector.get_support()
    feature_names = tfidf.get_feature_names()
    features_selected = np.array(feature_names)[support]
    features_selected = [i for i in features_selected]
    
    pickle.dump(tfidf, open(prediction_path+"/tfidf_vec"+str(k)+".pickle", "wb"))
    pickle.dump(chi2_selector, open(prediction_path+"/chi2_selector_tfidf"+str(k)+".pickle", "wb"))
    pickle.dump(features_selected, open(prediction_path+"/feature_selected_"+str(k)+".pickle", "wb"))
  else:
    features_selected = []
    pickle.dump(hashing, open(prediction_path+"/hashing_vec"+str(k)+".pickle", "wb"))
    pickle.dump(chi2_selector, open(prediction_path+"/chi2_selector_tfidf"+str(k)+".pickle", "wb"))
  return X,y,features_selected


def model_train_save(X,y, features_selected,prediction_path,phrase_dict,k,calibrated,ET,upsampling=False,SMOTE_Sampling=False, standardize=True):
  if ET == True:
    prediction_path = "tax_related/RF/Prediction/ET"
    
  train_sample_size = int(0.8*len(y))
  indices = list(range(len(y)))
  X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X, y, indices, train_size = train_sample_size, random_state=0,stratify=y) 


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
                                                                                     
  ## Build a Random Forest model

  print("Calibrated: " + str(calibrated))
  if calibrated == False:
    RF = RandomForestClassifier(oob_score= True,warm_start = False, class_weight = "balanced", random_state=0)
    ET_Classifier = ExtraTreesClassifier(random_state = 0)
  
  ## Set parameters for Grid Search

    n_estimators = [500, 800, 1000]
    max_depth = [35, 50]
    max_features=[35, 50]
    criterion=['entropy']
    hyperparameters = dict(criterion=criterion, n_estimators = n_estimators, max_depth = max_depth,max_features=max_features)
    
    if ET == True:
      ensemble = ET_Classifier
      print("Extra Tree")
    else:
      ensemble = RF
      print("Random Forest")
      
    clf = GridSearchCV(ensemble, hyperparameters, cv=5, verbose=2, n_jobs=-1, pre_dispatch = 32)
    
    logging.info("Grid Search Done")
    
    clf.fit(X_train, y_train)
    best_model = clf.best_estimator_
  
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
  
  
  ## Print best parameters
  if calibrated == False:
    print('Criterion: ' + str(best_model.get_params()['criterion']))
    print('max_depth: ' + str(best_model.get_params()['max_depth']))
    print('n_estimator: ' + str(best_model.get_params()['n_estimators']))
    print('max_features: '  + str(best_model.get_params()['max_features']))
    
  y_test_predict = best_model.predict(X_test)
  y_train_predict = best_model.predict(X_train)
  y_prob = best_model.predict_proba(X_test)
  y_prob_train = best_model.predict_proba(X_train)

  ## Print metrics
  rf_y, rf_x = calibration_curve(y_test, y_prob[:,1], n_bins=20,strategy='uniform')
  calibration_plot(rf_x,rf_y, y_prob[:,1], 'Random Forest',prediction_path,calibrated=True)
  print('Calibration plot created')
  train_acc = accuracy_score(y_train, y_train_predict)
  test_acc = accuracy_score(y_test, y_test_predict)
  print("Train Accuracy: " + str(train_acc))
  print("Test Accuracy: " + str(test_acc))
  
  f1_train = f1_score(y_train, y_train_predict,labels=y_train)
  f1_test = f1_score(y_test, y_test_predict,labels=y_test)
  print("Train F1: " + str(f1_train))
  print("Test F1: " + str(f1_test))
  
  precision_train = precision_score(y_train, y_train_predict,labels=y_train)
  precision_test = precision_score(y_test, y_test_predict,labels=y_test)
  print("Train precision: " + str(precision_train))
  print("Test precision: " + str(precision_test))
  
  recall_train = recall_score(y_train, y_train_predict,labels=y_train)
  recall_test = recall_score(y_test, y_test_predict,labels=y_test)
  print("Train recall: " + str(recall_train))
  print("Test recall: " + str(recall_test))
  
  aucroc = roc_auc_score(y_test, y_prob[:,1])
  aucroc_train = roc_auc_score(y_train, y_prob_train[:,1])
  print("Train AURCROC: "+str(aucroc_train))
  print("Test AUCROC: " + str(aucroc))
  
  tn,fp,fn,tp = confusion_matrix(y_test,y_test_predict).ravel()
  confusion_matrix_result = "Confusion matrix"+"\n"+ "True negatives: "+str(tn)+"\n"+"False positives: "+str(fp)+"\n"+"False negatives: "+str(fn)+"\n"+"True positives: "+str(tp)+"\n"
  print(confusion_matrix_result)
  
  
    
  if calibrated == False and len(features_selected)>1:
    importance = best_model.feature_importances_
    ranking = np.argsort(importance)[::-1]
    features_sorted = np.array(features_selected)[ranking]
    features = [i for i in features_sorted]   
    
    print("Top 50 features: ", features[0:50])
    print("Top 50 features magnitude percentage", importance[0:50]/np.sum(importance) )
    
  
  logging.info("Results printed")
  
  if standardize == True:
    append = 'standardized'
  else:
    append = ''
  
  if calibrated == False:
    pickle.dump(clf,open(prediction_path+"/best_model_no_calibration_"+str(k)+append+".pickle","wb"))
  else:
    pickle.dump(best_model,open(prediction_path+"/best_model_tfidf_all_calibrator_"+str(k)+append+".pickle","wb"))
  
  
  logging.info("model saved")
  
  if upsampling == True:
    file_name = prediction_path+'/result_RF_all_calibrator'+str(k)+'_upsampled'+append+'.txt'
    if  ET == True:
      file_name = prediction_path +'/result_ET_all_calibrator'+str(k)+append+'.txt'
  elif SMOTE_Sampling == True:
    file_name = prediction_path+'/result_RF_all_calibrator'+str(k)+'_SMOTE'+append+'.txt'

  else:
    file_name = prediction_path+'/result_RF_all_calibrator'+str(k)+append+'.txt'
  
  
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
    


def main():
  ## Load data from saved pickle files
  
  path = "/tax_related"
  prediction_path = "/tax_related/RF/Prediction"
  os.chdir(path)
  data = pickle.load(open('df_lexis.pickle',"rb")) # a list 
  
  logging.info("Data loaded")
    
  phrase2int = pd.read_pickle('/model_input/statutes-phrase2int.pkl')              
  phrase_dict = {value:key for key,value in phrase2int.items()}
  stop_words = text.ENGLISH_STOP_WORDS

          
  k=30000
    
    
  features_return = create_features(data=data,k=k,prediction_path=prediction_path,both=1, phrase_dict = phrase_dict, standardize = False)
          
  X=features_return[0]
  y=features_return[1]
  features_selected = features_return[2]
  model_train_save(X=X,y=y,features_selected=features_selected,prediction_path=prediction_path,phrase_dict=phrase_dict,k=k,ET=False,calibrated=True,upsampling=False, standardize= False)

    

if __name__ == "__main__":
    main()




