import pandas as pd
import numpy as np
import itertools
import os
import re
import logging 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from ast import literal_eval  


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
from sklearn.metrics import multilabel_confusion_matrix
import pickle
from sklearn.calibration import CalibratedClassifierCV

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

from imblearn.over_sampling import SMOTE


## Set logging format
logging.basicConfig(format='%(asctime)s  %(message)s', 
    datefmt='%y-%m-%d %H:%M:%S', level=logging.INFO)


def calibration_plot(x,y,name,prediction_path,calibrated):
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
  os.chdir(prediction_path)
  if calibrated == True:
    fig.savefig('Calibration_Plot_Calibrated_'+str(name)+'.pdf')
  else:
    fig.savefig('Calibration_Plot_'+str(name)+'.pdf')
    
    
    
def create_features(data,k,phrase_dict,prediction_path):  
  phrase= data['phrase'].values #[i[2] for i in data]
  dc_identifiers = data['dc_identifiers'].values #[i[0] for i in data]
  y = data['label_source'].values#[i[1] for i in data] # label
  phrase_lst = []

  for i in phrase:
    w=' '
    for j in i:
      w+= phrase_dict[j]+' '
    phrase_lst.append(w)

  tfidf = pickle.load(open('/model_input/tfidf_both_50000.pickle','rb'))
  X = tfidf.fit_transform(phrase_lst)

  del phrase,data
  gc.collect()

  logging.info("Features created")

## Select features using Chi sqaured

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


def model_train_save(X,y, features_selected,prediction_path,phrase_dict,k, calibrated,upsampling=False,SMOTE_Sampling=False,ET=False):

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
                                                                                     
## Buile a Random Forest model

  
  RF = RandomForestClassifier(oob_score= True,warm_start = False, class_weight = "balanced", random_state=0)
  ET_Classifier = ExtraTreesClassifier(random_state = 0)

## Set parameters for Grid Search
  
  n_estimators = [500,800]
  max_depth = [50,35]
  max_features=[20]
  criterion=['entropy']
  hyperparameters = dict(criterion=criterion, n_estimators = n_estimators, max_depth = max_depth,max_features=max_features)
  
  if ET == True:
    ensemble = ET_Classifier
    print("Extra Tree")
  else:
    ensemble = RF
    print("Random Forest")
    
  if calibrated == False:
    clf = GridSearchCV(ensemble, hyperparameters, cv=5, verbose=2, n_jobs=-1, pre_dispatch = 32)
    
    logging.info("Grid Search Done")
    
    clf.fit(X_train, y_train)
    best_model = clf.best_estimator_
  
    
  if calibrated == True:
    #X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size = int(0.8*train_sample_size), random_state=0,stratify=y_train) 
    if ET == False:
      model = RandomForestClassifier(random_state = 0, max_features=50, n_estimators=500, max_depth=100, class_weight = "balanced" )
    else:
      model = ExtraTreesClassifier(random_state = 0, max_features=50, n_estimators=500, max_depth=100, class_weight = "balanced" )
      
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
  
  for j in range(0,len(np.unique(y))):
    rf_y, rf_x = calibration_curve([1 if i==np.unique(y)[j] else 0 for i in y_test], y_prob[:,j], n_bins=10)
    calibration_plot(rf_x,rf_y,'Random Forest:'+np.unique(y)[j],prediction_path,calibrated=True)
  
  
  
  ## Print metrics
  print(y_prob_train[0])
  train_acc = accuracy_score(y_train, y_train_predict)
  test_acc = accuracy_score(y_test, y_test_predict)
  print("Train Accuracy: " + str(train_acc))
  print("Test Accuracy: " + str(test_acc))
  
  f1_train = f1_score(y_train, y_train_predict,labels=y_train,average='macro')
  f1_test = f1_score(y_test, y_test_predict,labels=y_test,average='macro')
  print("Train F1: " + str(f1_train))
  print("Test F1: " + str(f1_test))
  
  precision_train = precision_score(y_train, y_train_predict,labels=y_train,average='macro')
  precision_test = precision_score(y_test, y_test_predict,labels=y_test,average='macro')
  print("Train precision: " + str(precision_train))
  print("Test precision: " + str(precision_test))
  
  recall_train = recall_score(y_train, y_train_predict,labels=y_train,average='macro')
  recall_test = recall_score(y_test, y_test_predict,labels=y_test,average='macro')
  print("Train recall: " + str(recall_train))
  print("Test recall: " + str(recall_test))
  confusion_matrix_result = confusion_matrix(y_test,y_test_predict,labels=list(np.unique(y)))
  confusion_matrix_multi = multilabel_confusion_matrix(y_test, y_test_predict,labels=list(np.unique(y)))
  print(confusion_matrix_result)
  print(confusion_matrix_multi)
  
  if calibrated == False:
    importance = best_model.feature_importances_
    ranking = np.argsort(importance)[::-1]
    features_sorted = np.array(features_selected)[ranking]
    features = [phrase_dict[i] for i in features_sorted]
  
    print("Top 20 features: ", features[0:20])
    print("Top 20 features magnitude percentage", importance[0:20]/np.sum(importance) )
  
  logging.info("Results printed")
  
  
  pickle.dump(best_model,open(prediction_path+"/best_model_tfidf"+str(k)+".pickle","wb"))
  if calibrated == False:
    pickle.dump(clf,open(prediction_path+"/clf_tfidf"+str(k)+".pickle","wb"))
    
  
  logging.info("model saved")
  
  if upsampling == True:
    file_name = prediction_path+'/result_RF_'+str(k)+'_upsampled'+'.txt'
    if  ET == True:
      file_name = prediction_path +'/result_ET_'+str(k)+'.txt'
  elif SMOTE_Sampling == True:
    file_name = prediction_path+'/result_RF_'+str(k)+'_SMOTE'+'.txt'

  else:
    file_name = prediction_path+'/result_RF_'+str(k)+'.txt'
  
  
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
    print(confusion_matrix_result,file=text_file)
    print(confusion_matrix_multi,file=text_file)
    print(classification_report(y_test, y_test_predict),file=text_file)
    if calibrated == False:
      print(features[0:20],file=text_file)
      print(importance[0:20]/np.sum(importance),file=text_file )
    


def main():
  ## Load data from saved pickle files
  
  path = "/tax_source/"
  prediction_path = "/Prediction"
  data = pd.read_csv('df_final_new.csv',index_col=0)
  data = data.replace(np.nan, 'Other', regex=True)
  #data = data[data['label'] != '']
  data.phrase=data['phrase'].apply(literal_eval)
  
  
  logging.info("Data loaded")
    
  phrase2int = pd.read_pickle('/model_input/statutes-phrase2int.pkl')              
  phrase_dict = {value:key for key,value in phrase2int.items()}
  stop_words = text.ENGLISH_STOP_WORDS

## Remove phrases still in stop_words list
  
  for i in range(0,data.shape[0]):
    for j in data['phrase'].iloc[i]:
        if phrase_dict[j] in stop_words:
          data['phrase'].values[i] = list(filter((j).__ne__, data['phrase'].iloc[i]))
          
  k=30000
    
    
  features_return = create_features(data=data,k=k,phrase_dict = phrase_dict,prediction_path=prediction_path)
          
  X=features_return[0]
  y=features_return[1]
  features_selected = features_return[2]
  model_train_save(X=X,y=y,features_selected=features_selected,prediction_path=prediction_path,phrase_dict=phrase_dict,k=k,calibrated = True)

    

if __name__ == "__main__":
    main()

