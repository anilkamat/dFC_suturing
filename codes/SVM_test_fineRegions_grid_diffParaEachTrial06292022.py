from math import gamma
from operator import index
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, plot_confusion_matrix
from sklearn.svm import SVC
from Rank_variables_SVM_accuracy import rankVariable
#PATH = r'C:\Users\_Kamat_\Desktop\RPI\ResearchWork\Papers_\Effective_Connectivity\EEG_fNIRS_paper_Brain_informatics\fNIRS_codes_results\Results\Results_May31_region_level\Connectivities'
#PATH2 = r'C:\Users\_Kamat_\Desktop\RPI\ResearchWork\Papers_\Effective_Connectivity\EEG_fNIRS_paper_Brain_informatics\fNIRS_codes_results\Results\Results_May31_region_level\Connectivities\SVM_results'
PATH = r'C:\Users\_Kamat_\Desktop\RPI\ResearchWork\Papers_\Effective_Connectivity\EEG_fNIRS_paper_Brain_informatics\fNIRS_codes_results\Results\Results_Jun20_fineRegions\Connectivities'
PATH2 = r'C:\Users\_Kamat_\Desktop\RPI\ResearchWork\Papers_\Effective_Connectivity\EEG_fNIRS_paper_Brain_informatics\fNIRS_codes_results\Results\Results_Jun20_fineRegions\SVM_Results'
SC = pd.read_csv(r'C:\Users\_Kamat_\Desktop\RPI\ResearchWork\Papers_\Effective_Connectivity\EEG_fNIRS_paper_Brain_informatics\fNIRS_codes_results\Codes\SC_vector_06282022.csv', header = 0, index_col=None, sep=',')
SC_existing = SC.loc[:, (SC.any() > 0)]
connections = SC_existing.columns[:]


filenames = []
Data_all = [] # all subjects, all trial, all HTA
data = {}
fnames = []
for filename in os.listdir(PATH):
    if filename.endswith("csv"): 
        df = pd.read_csv(os.path.join(PATH, filename), header = 0, index_col=False, sep=',')
        df = df[(df != 0).all(1)]  # remove rows with zeros.
        df = df[connections]
        Data_all.append(df)
        filenames.append(filename)
        fname = filename.strip('fNIRS_GC_Exp_Nov')
        fname = fname.strip('.cs')
        fnames.append(fname)
        data[fname] = df
Test_data = Data_all
Data_all = pd.concat(Data_all)
#connections = df.columns[0:132]
# test_acc = {}
# for connection in connections:
#   test_acc[connection] = []
# baseName = '/content/gdrive/MyDrive/data_used_colab/EEG_fNIRS_suturing/'
def SVM(X_train,y_train):
  svclassifier.fit(X_train, y_train)
  #y_pred_test = svclassifier.predict(X_test)
  y_pred_train = svclassifier.predict(X_train)
  #print(y_pred)
  #print(confusion_matrix(y_test,y_pred_test))
  #print(classification_report(y_test,y_pred_test))
#   plot_confusion_matrix(svclassifier, X_test, y_test)
#   plt.show()
  # on train data
  #print(confusion_matrix(y_train,y_pred_train))
  metrics = classification_report(y_train,y_pred_train)
  #print(metrics)
  acc_train = accuracy_score(y_train,y_pred_train)
  #acc_test = accuracy_score(y_test,y_pred_test)
  #print('The accuracy',acc)
  return acc_train

def SVM_test(X_test,y_test): # for test datasets
  y_pred_test = svclassifier.predict(X_test)
  #print(y_pred)
  #print(confusion_matrix(y_test,y_pred_test))
  #print(classification_report(y_test,y_pred_test))
  #plot_confusion_matrix(svclassifier, X_test, y_test)  
  acc_test = accuracy_score(y_test,y_pred_test)
  #print('The accuracy',acc)
  return acc_test
def plots(acc, connection, plt_title):
    n = np.arange(len(connection))
    fig = plt.figure(figsize = (8, 4))
    # creating the bar plot
    plt.bar( connection, acc, color ='maroon', width = 0.2)
    plt.xlabel("Leave one variables out")
    plt.ylabel("SVM Accuracy")
    plt.title(plt_title)
    ext2 ='.png'
    plt_title = plt_title +ext2
    plt.xticks(rotation=45)
    for i in range(len(acc)):
        plt.annotate(str(acc[i]), xy=(n[i],acc[i]), ha='center', va='bottom')
    #dd_value_label(acc)
    plt.tight_layout()
    #plt.savefig(plt_title)
    #files.download(plt_title) 
    plt.show()
    plt.close('all')

#train on data from all HTA and trials
X = Data_all.iloc[:,:-1]
y = Data_all.iloc[:,-1]
trials = ['Tr_1','Tr_2','Tr_3']
model_para_grid = {
  'c' :[0.0001, 0.001, 0.01, 0.1, 1, 10, 50,100,500,1000,10000],
  'gamma':[0.0001,0.001, 0.005,0.01,0.05, 0.1,1,10,100,1000],
  'kernel' : ['linear','rbf','sigmoid', 'poly']
}
best_model_T1 = {
  'c' :[],
  'gamma':[],
  'kernel' : []
}
best_model_T2 = {
  'c' :[],
  'gamma':[],
  'kernel' : []
}
best_model_T3 = {
  'c' :[],
  'gamma':[],
  'kernel' : []
}
best_acc_test1 = 0
best_acc_test2 = 0
best_acc_test3 = 0
for kernel_ in model_para_grid['kernel']:
  for gamma_ in model_para_grid['gamma']:
    for c_ in model_para_grid['c']:
      if kernel_ == 'linear':
        svclassifier = SVC(kernel=kernel_, C=c_)#, gamma=100, C=0.1,random_state=1,) # ,
      elif kernel_ == 'poly':
        svclassifier = SVC(kernel=kernel_, gamma=gamma_, C=c_, degree = 8)# gamma=100, C=0.1,random_state=1,) # ,
      else:
        svclassifier = SVC(kernel=kernel_, gamma=gamma_, C=c_)# gamma=100, C=0.1,random_state=1,) # ,
      test_acc = {}
      for trial in trials: # hold out one trial (all subtasks) and train on remainings
        Train_data = []
        Test_data = []
        for names in fnames:
          df1 = data[names]
          if names[0:4] == trial:
            Test_data.append(df1)
          else:
            Train_data.append(df1)
        Train_data = pd.concat(Train_data)
        # test_acc ={}
        for connection in connections[0:46]:
          Train_data_dropped = Train_data.drop(connection, axis = 1, inplace=False)
          X_train = Train_data_dropped.iloc[:,:-1]
          y_train = Train_data_dropped.iloc[:,-1]
          mean = X_train.mean(axis = 0)
          std = X_train.std( axis = 0)
          X_train = (X_train-mean)/std
          #print('normalized Xtrain :', X_train)

          acc_train = SVM(X_train,y_train)
          #print(f'Train acc:{acc_train}')
          accuracies = []
          test_acc[connection] = []
          for i in range(4):
            Test_data_dropped =  Test_data[i].drop(connection, axis = 1, inplace=False)
            X_test = Test_data_dropped.iloc[:,:-1]
            y_test = Test_data_dropped.iloc[:,-1]
            X_test = (X_test-mean)/std

            acc_test = SVM_test(X_test,y_test)
            accuracies.append(acc_test)
            if (trial == 'Tr_1') and (acc_test > best_acc_test1) and i==0:
              best_acc_test1 = acc_test
              best_model_T1['c'] = c_
              best_model_T1['gamma'] = gamma_
              best_model_T1['kernel']= kernel_
              best_model_T1['test_acc'] = acc_test
              print('best model so far-Trial 1: ',best_model_T1)
            if (trial == 'Tr_2') and (acc_test > best_acc_test2) and i==0:
              best_acc_test2 = acc_test
              best_model_T2['c'] = c_
              best_model_T2['gamma'] = gamma_
              best_model_T2['kernel']= kernel_
              best_model_T2['test_acc'] = acc_test      
              print('best model so far-Trial 2: ',best_model_T2)        
            if (trial == 'Tr_3') and (acc_test > best_acc_test3) and i==0:
              best_acc_test3 = acc_test
              best_model_T3['c'] = c_
              best_model_T3['gamma'] = gamma_
              best_model_T3['kernel']= kernel_
              best_model_T3['test_acc'] = acc_test
              print('best model so far-Trial 3: ',best_model_T3)
          #   #print(accuracies)
          # if not isinstance(test_acc[connection],list):
          #   test_acc[connection] = [test_acc[connection]]
          # test_acc[connection].append(accuracies)
        #print(test_acc)
print(best_model_T3)

test_acc = {}
for trial in trials: # hold out one trial (all subtasks) and train on remainings
  if trial =='Tr_1':
    svclassifier = SVC(kernel=best_model_T1['kernel'], gamma = best_model_T1['gamma'], C=best_model_T1['c'],random_state=1)#, gamma=100, C=0.1) # ,
  if trial =='Tr_2':
    svclassifier = SVC(kernel=best_model_T2['kernel'], gamma = best_model_T2['gamma'], C=best_model_T2['c'],random_state=1)#, gamma=100, C=0.1) # ,
  if trial =='Tr_3':
    svclassifier = SVC(kernel=best_model_T3['kernel'], gamma = best_model_T3['gamma'], C=best_model_T3['c'],random_state=1)#, gamma=100, C=0.1) # , 
  Train_data = []
  Test_data = []
  for names in fnames:
    df1 = data[names]
    if names[0:4] == trial:
      Test_data.append(df1)
    else:
      Train_data.append(df1)
  Train_data = pd.concat(Train_data)
  # test_acc ={}
  for connection in connections[0:46]:
    Train_data_dropped = Train_data.drop(connection, axis = 1, inplace=False)
    X_train = Train_data_dropped.iloc[:,:-1]
    #print('size of X_train after dropp: ',X_train.shape)
    y_train = Train_data_dropped.iloc[:,-1]
    mean = X_train.mean(axis = 0)
    std = X_train.std( axis = 0)
    X_train = (X_train-mean)/std
    acc_train = 0.0
    acc_train = SVM(X_train,y_train)
    #print(f'Train acc:{acc_train}')
    accuracies = []
    test_acc[connection] = []
    for i in range(4):
      Test_data_dropped =  Test_data[i].drop(connection, axis = 1, inplace=False)
      X_test = Test_data_dropped.iloc[:,:-1]
      y_test = Test_data_dropped.iloc[:,-1]
      X_test = (X_test-mean)/std
      acc_test = SVM_test(X_test,y_test)
      accuracies.append(acc_test)
     
    test_acc[connection] = accuracies
    #   #print(accuracies)
    # if not isinstance(test_acc[connection],list):
    #   test_acc[connection] = [test_acc[connection]]
    # test_acc[connection].append(accuracies)
  #print(test_acc)
  df2 = pd.DataFrame(test_acc)
  df2 = df2.T
  FullFilename = os.path.join(PATH2,trial+'.csv')
  df2.to_csv(FullFilename)

# Test with all the features/connectivities.
test_acc_all = {}  # Calculates the base accuracy i.e. with all the connectivities
for trial in trials: # hold out one trial (all subtasks) and train on remainings
  if trial =='Tr_1':
    svclassifier = SVC(kernel=best_model_T1['kernel'], gamma = best_model_T1['gamma'], C=best_model_T1['c'],random_state=1)#, gamma=100, C=0.1) # ,
  if trial =='Tr_2':
    svclassifier = SVC(kernel=best_model_T2['kernel'], gamma = best_model_T2['gamma'], C=best_model_T2['c'],random_state=1)#, gamma=100, C=0.1) # ,
  if trial =='Tr_3':
    svclassifier = SVC(kernel=best_model_T3['kernel'], gamma = best_model_T3['gamma'], C=best_model_T3['c'],random_state=1)#, gamma=100, C=0.1) # , 
  Train_data = []
  Test_data = []
  for names in fnames:
    df1 = data[names]
    if names[0:4] == trial:
      Test_data.append(df1)
    else:
      Train_data.append(df1)
  Train_data = pd.concat(Train_data)
# test_acc ={}
  X_train = Train_data.iloc[:,:-1]
  #print('size of X_train_all: ',X_train.shape)
  y_train = Train_data.iloc[:,-1]
  mean = X_train.mean(axis = 0)
  std = X_train.std( axis = 0)
  X_train = (X_train-mean)/std
  acc_train = 0.0
  acc_train = SVM(X_train,y_train)
  #print(f'Train acc:{acc_train}')
  accuracies_all = []
  test_acc_all[connection] = []
  for i in range(4):
    X_test_all = Test_data[i].iloc[:,:-1]
    y_test_all = Test_data[i].iloc[:,-1]
    X_test_all = (X_test_all-mean)/std
    acc_test = SVM_test(X_test_all,y_test_all)
    accuracies_all.append(acc_test)
  test_acc_all[connection] = accuracies_all

  df3 = pd.DataFrame(test_acc_all)
  df3 = df3.T
  #df2.index = connection
  #df2.columns = i
  #df1 = df1.transpose()
  FullFilename2 = os.path.join(PATH2,trial+'all_06292022'+'.csv')
  df3.to_csv(FullFilename2)

trialandST = ['T1ST1','T1ST2','T1ST3',	'T1ST4',	'T2ST1',	'T2ST2',	'T2ST3','T2ST4','T3ST1','T3ST2','T3ST3','T3ST4']
PATH3 = r'C:\Users\_Kamat_\Desktop\RPI\ResearchWork\Papers_\Effective_Connectivity\EEG_fNIRS_paper_Brain_informatics\fNIRS_codes_results\Results\Results_Jun20_fineRegions\SVM_Results'
filename1 = os.path.join(PATH3,'Results_06292022.xlsx')
# ranked = rankVariable(filename1)