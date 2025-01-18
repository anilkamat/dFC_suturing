from ast import Mod
from calendar import c
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, matthews_corrcoef, balanced_accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut
import pandas as pd
import pickle
from chEEG_plot_mesh_grid_06162023 import plot_ROC

def grid_search( data, config_subtask, config, set_number,savePath_results, Type = 'add_connectivity'):
    connectivity_set =  list(data)

    model_para_grid = config['model_para_grid']
    para_model_ = {}
    para_model_with_mcc = {}
    para_model_with_bas = {}
    best_model_ = {}
    Phase = config_subtask['phase']
    trial= config_subtask['trial'] #[0:1]
    if Type == 'all_connectivity':
        ModelName = 'all_connectivity_'+trial+Phase
        para_model_[ModelName]=np.zeros(((len(model_para_grid['gamma']), len(model_para_grid['c']))))
        para_model_with_mcc[ModelName]=np.zeros(((len(model_para_grid['gamma']), len(model_para_grid['c']))))
        para_model_with_bas[ModelName]=np.zeros(((len(model_para_grid['gamma']), len(model_para_grid['c']))))
        best_model_[ModelName+'c'] = 0.0
        best_model_[ModelName+'gamma'] = 0.0
        best_model_[ModelName+'kernel'] = ''
        best_model_[ModelName+'test_acc_CV'] =0.0
        best_model_[ModelName+'test_mcc_CV'] =0.0
        best_model_[ModelName+'test_bas_CV'] =0.0
    else:
        ModelName = 'add_connectivity_'+trial+Phase
        para_model_[ModelName]=np.zeros(((len(model_para_grid['gamma']), len(model_para_grid['c']))))
        para_model_with_mcc[ModelName]=np.zeros(((len(model_para_grid['gamma']), len(model_para_grid['c']))))
        para_model_with_bas[ModelName]=np.zeros(((len(model_para_grid['gamma']), len(model_para_grid['c']))))
        best_model_[ModelName+'c'] = 0.0
        best_model_[ModelName+'gamma'] = 0.0
        best_model_[ModelName+'kernel'] = ''
        best_model_[ModelName+'test_acc_CV'] =0.0
        best_model_[ModelName+'test_mcc_CV'] =0.0
        best_model_[ModelName+'test_bas_CV'] =0.0
    for kernel_ in model_para_grid['kernel_']:
        n = 0
        print(f'#### Kernel: {kernel_} ####')
        for gamma_ in model_para_grid['gamma']:
            o = 0
            for c_ in model_para_grid['c']:
                svclassifier = []  #Resets classifier over loops
                if kernel_ == 'linear':
                    svclassifier = SVC(kernel=kernel_,random_state=1)
                else:
                    svclassifier = SVC(kernel=kernel_, gamma=gamma_, C=c_,random_state=1)
                if Type=='all_connectivity':
                    X = data.iloc[:,:-1]
                    # print('X.shape: ',X.shape)
                    Y = data.iloc[:,-1]
                    mean = X.mean(axis = 0)
                    std = X.std( axis = 0)
                    X = (X-mean)/std
                    acc_test_CV = []
                    mcc_test_CV = []
                    bas_test_CV = []
                    loo = LeaveOneOut()
                    loo.get_n_splits(X)
                    y_test_cv = []
                    y_test_pred_cv = []
                    for train_indx, test_indx in loo.split(X, Y):
                        x_train,x_test = X.iloc[train_indx],X.iloc[test_indx]
                        y_train,y_test = Y.iloc[train_indx],Y.iloc[test_indx]
                        svclassifier.fit(x_train, y_train)
                        y_pred_test = svclassifier.predict(x_test)
                        y_test_pred_cv.append(y_pred_test)
                        y_test_cv.append(y_test)
                    y_test_cv = np.hstack(y_test_cv)
                    y_test_pred_cv = np.hstack(y_test_pred_cv)
                    acc_test_CV = accuracy_score(y_test_cv,y_test_pred_cv)
                    mcc_test_CV = matthews_corrcoef(y_test_cv,y_test_pred_cv)
                    bas_test_CV = balanced_accuracy_score(y_test_cv,y_test_pred_cv)
                    ModelName = 'all_connectivity_'+trial+Phase

                    if (acc_test_CV > best_model_[ModelName+'test_acc_CV']) :
                        #pickle.dump(svclassifier, open(ModelName+'.sav', 'wb'))     #save the best model so far
                        best_model_[ModelName+'c'] = c_
                        best_model_[ModelName+'gamma'] = gamma_
                        best_model_[ModelName+'kernel']= kernel_
                        best_model_[ModelName+'test_acc_CV'] =acc_test_CV
                        best_model_[ModelName+'test_mcc_CV'] =mcc_test_CV
                        best_model_[ModelName+'test_bas_CV'] =bas_test_CV
                        
                        if mcc_test_CV > 0.40:
                            print('best test mcc:',mcc_test_CV)
                            print(f'best parameters n: {n}, o: {o}, gamma_:{gamma_}, c:{c_}')
                    # store all the parameters and corrosponding accuracies
                    para_model_[ModelName][n,o] = acc_test_CV
                    para_model_with_mcc[ModelName][n,o] = mcc_test_CV
                    para_model_with_bas[ModelName][n,o] = bas_test_CV

                if Type=='add_connectivity':

                    X = data.iloc[:,:-1]
                    Y = data.iloc[:,-1]
                    mean = X.mean(axis = 0)
                    std = X.std( axis = 0)
                    X = (X-mean)/std
                    loo = LeaveOneOut()
                    loo.get_n_splits(X)
                    acc_test_CV = []
                    mcc_test_CV = []
                    bas_test_CV = []
                    y_test_cv = []
                    y_test_pred_cv = []

                    for train_indx, test_indx in loo.split(X, Y):
                        x_train,x_test = X.iloc[train_indx],X.iloc[test_indx]
                        y_train,y_test = Y.iloc[train_indx],Y.iloc[test_indx]
                        svclassifier.fit(x_train, y_train)
                        y_pred_test = svclassifier.predict(x_test)
                        y_test_pred_cv.append(y_pred_test)
                        y_test_cv.append(y_test)
                    y_test_cv = np.hstack(y_test_cv)
                    y_test_pred_cv = np.hstack(y_test_pred_cv)
                    acc_test_CV = accuracy_score(y_test_cv,y_test_pred_cv)
                    mcc_test_CV = matthews_corrcoef(y_test_cv,y_test_pred_cv)
                    bas_test_CV = balanced_accuracy_score(y_test_cv,y_test_pred_cv)
                    #acc_test_CV_std = np.std(acc_test_CV)
                    #print(f'Train acc:{acc_train}')
                    ModelName = 'add_connectivity_'+trial+Phase
                    #print('model_name, mcc_test_CV_mean:',ModelName, mcc_test_CV_mean)
                    #para_model_T1_[ModelName[:-4]] = np.zeros((20,7,8))   # (connection, gamma, c)
                    if (acc_test_CV > best_model_[ModelName+'test_acc_CV']) :
                        pickle.dump(svclassifier, open(ModelName+'.sav', 'wb'))     #save the best model so far
                        best_model_[ModelName+'c'] = c_
                        best_model_[ModelName+'gamma'] = gamma_
                        best_model_[ModelName+'kernel']= kernel_
                        best_model_[ModelName+'test_acc_CV'] =acc_test_CV
                        best_model_[ModelName+'test_mcc_CV'] =mcc_test_CV
                        best_model_[ModelName+'test_bas_CV'] =bas_test_CV
                        if mcc_test_CV > 0.40:
                            print('best test mcc:', mcc_test_CV)
                            print(f'best parameters n: {n}, o: {o}, gamma_:{gamma_}, c:{c_}')
                    # store all the parameters and corrosponding accuracies
                    para_model_[ModelName][n,o] = acc_test_CV
                    para_model_with_mcc[ModelName][n,o] = mcc_test_CV
                    para_model_with_bas[ModelName][n,o] = bas_test_CV
                if kernel_ == 'linear':
                    break
                o+=1
            if kernel_ == 'linear':
               break
            n +=1 
        if kernel_ == 'linear':
            break
    return para_model_,best_model_, para_model_with_mcc, para_model_with_bas

    