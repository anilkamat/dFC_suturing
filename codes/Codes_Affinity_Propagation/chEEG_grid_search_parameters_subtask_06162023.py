from ast import Mod
from calendar import c
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, matthews_corrcoef, balanced_accuracy_score
from sklearn.cluster import KMeans
from sklearn.model_selection import LeaveOneOut, KFold, StratifiedKFold
import pandas as pd
import pickle
from chEEG_plot_mesh_grid_06162023 import plot_ROC

def grid_search( data, config_subtask, config, n_folds, set_number,savePath_results, Type = 'add_connectivity'):
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
        best_model_[ModelName+'test_acc'] = 0.0
        best_model_[ModelName+'test_acc_CV'] =0.0
        best_model_[ModelName+'test_mcc_CV'] =0.0
        best_model_[ModelName+'test_bas_CV'] =0.0
    else:
        for m in range(20):  # for feature/connectivity add datasets.
            ModelName = 'add_connectivity_'+trial+Phase
            #print('modelname:',ModelName)
            para_model_[ModelName]=np.zeros(((len(model_para_grid['gamma']), len(model_para_grid['c']))))
            para_model_with_mcc[ModelName]=np.zeros(((len(model_para_grid['gamma']), len(model_para_grid['c']))))
            para_model_with_bas[ModelName]=np.zeros(((len(model_para_grid['gamma']), len(model_para_grid['c']))))
            best_model_[ModelName+'c'] = 0.0
            best_model_[ModelName+'gamma'] = 0.0
            best_model_[ModelName+'kernel'] = ''
            best_model_[ModelName+'test_acc'] = 0.0
            best_model_[ModelName+'test_acc_CV'] =0.0
            best_model_[ModelName+'test_mcc_CV'] =0.0
            best_model_[ModelName+'test_bas_CV'] =0.0
    test_acc = []
    cv = StratifiedKFold(n_splits=n_folds)
    for kernel_ in model_para_grid['kernel_']:
        n = 0
        print(f'#### Kernel: {kernel_} ####')
        for gamma_ in model_para_grid['gamma']:
            o = 0
            for c_ in model_para_grid['c']:
                APclassifier = []  #Resets classifier over loops
                APclassifier = KMeans(n_clusters=2)
                temp_ = np.zeros((20))
                if Type=='all_connectivity':
                    X = data.iloc[:,:-1]
                    # print('X.shape: ',X.shape)
                    Y = data.iloc[:,-1]
                    mean = X.mean(axis = 0)
                    std = X.std( axis = 0)
                    X = (X-mean)/std
                    #print('normalized Xtrain :', X_train)
                    # loo = LeaveOneOut()
                    # loo.get_n_splits(X)
                    # for train_indx, test_indx in loo.split(X):
                    #     #print(f'train_indx: {train_indx} test_indx: {test_indx}')
                    #     x_train,x_test = X.iloc[train_indx],X.iloc[test_indx]
                    #     y_train,y_test = Y.iloc[train_indx],Y.iloc[test_indx]
                    #     #print(f'x_train: {x_train}')
                    #     acc_train = SVM(x_train,y_train)
                    #     acc_test = SVM_test(x_test,y_test)
                    #     acc_test_CV.append(acc_test)
                    # acc_test_CV_mean = np.mean(acc_test_CV)
                    # kfold
                    kf = KFold(n_splits=n_folds)
                    kf.get_n_splits(X)
                    # loo = LeaveOneOut()
                    # loo.get_n_splits(X)
                    acc_test_CV = []
                    mcc_test_CV = []
                    bas_test_CV = []
                    cv = StratifiedKFold(n_splits=n_folds)
                    y_test_cv = []
                    y_test_pred_cv = []
                    for train_indx, test_indx in cv.split(X, Y):
                        #print(f'train_indx: {train_indx} test_indx: {test_indx}')
                        x_train,x_test = X.iloc[train_indx],X.iloc[test_indx]
                        y_train,y_test = Y.iloc[train_indx],Y.iloc[test_indx]
                        #acc_train = SVM(x_train,y_train, svclassifier)
                        APclassifier.fit(x_train, y_train)
                       
                        y_pred_test = APclassifier.predict(x_test)
                        for i in range(len(y_pred_test)):
                            if y_pred_test[i] ==0:
                                y_pred_test[i] = -1
                        #print('Confusion Mat: ', confusion_matrix(y_test,y_pred_test))
                        #print(classification_report(y_test,y_pred_test))
                        #plot_confusion_matrix(svclassifier, X_test, y_test)  
                        acc_test = accuracy_score(y_test,y_pred_test)
                        mcc_test = matthews_corrcoef(y_test,y_pred_test)
                        bas_test = balanced_accuracy_score(y_test,y_pred_test)

                        acc_test_CV.append(acc_test)
                        mcc_test_CV.append(mcc_test)
                        bas_test_CV.append(bas_test)
                        y_test_pred_cv.append(y_pred_test)
                        y_test_cv.append(y_test)
                    y_test_cv = np.hstack(y_test_cv)
                    y_test_pred_cv = np.hstack(y_test_pred_cv)
                    acc_test_CV_mean = np.mean(acc_test_CV)
                    mcc_test_CV_mean = np.mean(mcc_test_CV)
                    bas_test_CV_mean = np.mean(bas_test_CV)

                    #acc_test_CV_std = np.std(acc_test_CV)
                    #print(f'Train acc:{acc_train}')
                    ModelName = 'all_connectivity_'+trial+Phase
                    #print('model_name:',ModelName)
                    #para_model_T1_[ModelName[:-4]] = np.zeros((20,7,8))   # (connection, gamma, c)
                    if (acc_test_CV_mean > best_model_[ModelName+'test_acc_CV']) :
                        #pickle.dump(svclassifier, open(ModelName+'.sav', 'wb'))     #save the best model so far
                        best_model_[ModelName+'c'] = c_
                        best_model_[ModelName+'gamma'] = gamma_
                        best_model_[ModelName+'kernel']= kernel_
                        best_model_[ModelName+'test_acc'] = acc_test
                        best_model_[ModelName+'test_acc_CV'] =acc_test_CV_mean
                        best_model_[ModelName+'test_mcc_CV'] =mcc_test_CV_mean
                        best_model_[ModelName+'test_bas_CV'] =bas_test_CV_mean
                        
                        #plot_ROC(svclassifier, X,Y, n_folds,set_number,savePath_results, type = 'BaseLine')
                        # cm=confusion_matrix(y_test_cv,y_test_pred_cv)
                        #plot_CF_matrix(y_test_cv,y_test_pred_cv,svclassifier,connectivity_set,ModelName, \
                            #set_number, labels= ['SA','MA'], type = 'BaseLine')
                        if mcc_test_CV_mean > 0.40:
                            print('best test mcc:',mcc_test_CV_mean)
                            print(f'best parameters n: {n}, o: {o}, gamma_:{gamma_}, c:{c_}')
                    # store all the parameters and corrosponding accuracies
                    para_model_[ModelName][n,o] = acc_test_CV_mean
                    para_model_with_mcc[ModelName][n,o] = mcc_test_CV_mean
                    para_model_with_bas[ModelName][n,o] = bas_test_CV_mean
                    test_acc=best_model_[ModelName+'test_acc_CV']

                if Type=='add_connectivity':

                    X = data.iloc[:,:-1]
                    Y = data.iloc[:,-1]
                    mean = X.mean(axis = 0)
                    std = X.std( axis = 0)
                    X = (X-mean)/std
                    #print('normalized Xtrain :', X_train)
                    kf = KFold(n_splits=n_folds)
                    kf.get_n_splits(X)
                    # loo = LeaveOneOut()
                    # loo.get_n_splits(X)
                    acc_test_CV = []
                    mcc_test_CV = []
                    bas_test_CV = []
                    y_test_cv = []
                    y_test_pred_cv = []
                    cv = StratifiedKFold(n_splits=n_folds)
                    for train_indx, test_indx in cv.split(X, Y):
                        x_train,x_test = X.iloc[train_indx],X.iloc[test_indx]
                        y_train,y_test = Y.iloc[train_indx],Y.iloc[test_indx]
                        #acc_train = SVM(x_train,y_train, svclassifier)
                        APclassifier.fit(x_train, y_train)
                        y_pred_test = APclassifier.predict(x_test)
                        for i in range(len(y_pred_test)):
                            if y_pred_test[i] ==0:
                                y_pred_test[i] = -1
                        #print('Confusion Mat: ', confusion_matrix(y_test,y_pred_test))
                        #print(classification_report(y_test,y_pred_test))
                        #plot_confusion_matrix(svclassifier, X_test, y_test)  
                        acc_test = accuracy_score(y_test,y_pred_test)
                        mcc_test = matthews_corrcoef(y_test,y_pred_test)
                        bas_test = balanced_accuracy_score(y_test,y_pred_test)
                        #acc_test, mcc_test, bas_test = SVM_test(x_test,y_test, svclassifier)

                        acc_test_CV.append(acc_test)
                        mcc_test_CV.append(mcc_test)
                        bas_test_CV.append(bas_test)
                        y_test_pred_cv.append(y_pred_test)
                        y_test_cv.append(y_test)
                    y_test_cv = np.hstack(y_test_cv)
                    y_test_pred_cv = np.hstack(y_test_pred_cv)
                    acc_test_CV_mean = np.mean(acc_test_CV)
                    mcc_test_CV_mean = np.mean(mcc_test_CV)
                    bas_test_CV_mean = np.mean(bas_test_CV)
                    #acc_test_CV_std = np.std(acc_test_CV)
                    #print(f'Train acc:{acc_train}')
                    ModelName = 'add_connectivity_'+trial+Phase
                    #print('model_name, mcc_test_CV_mean:',ModelName, mcc_test_CV_mean)
                    #para_model_T1_[ModelName[:-4]] = np.zeros((20,7,8))   # (connection, gamma, c)
                    if (acc_test_CV_mean > best_model_[ModelName+'test_acc_CV']) :
                        pickle.dump(APclassifier, open(ModelName+'.sav', 'wb'))     #save the best model so far
                        best_model_[ModelName+'c'] = c_
                        best_model_[ModelName+'gamma'] = gamma_
                        best_model_[ModelName+'kernel']= kernel_
                        best_model_[ModelName+'test_acc'] = acc_test
                        best_model_[ModelName+'test_acc_CV'] =acc_test_CV_mean
                        best_model_[ModelName+'test_mcc_CV'] =mcc_test_CV_mean
                        best_model_[ModelName+'test_bas_CV'] =bas_test_CV_mean
                        #plot_ROC(svclassifier, X,Y, n_folds,set_number,savePath_results, type = 'AddConnectivity')
                        #plot_CF_matrix(y_test_cv,y_test_pred_cv,svclassifier,connectivity_set,ModelName, \
                            #set_number, labels= ['SA','MA'], type = 'AddConnectivity')
                        if mcc_test_CV_mean > 0.40:
                            print('best test mcc:', mcc_test_CV_mean)
                            print(f'best parameters n: {n}, o: {o}, gamma_:{gamma_}, c:{c_}')
                    # store all the parameters and corrosponding accuracies
                    para_model_[ModelName][n,o] = acc_test_CV_mean
                    para_model_with_mcc[ModelName][n,o] = mcc_test_CV_mean
                    para_model_with_bas[ModelName][n,o] = bas_test_CV_mean
                    test_acc=best_model_[ModelName+'test_acc_CV']
                    # print('best Mean_test acc:', test_acc)
                if kernel_ == 'linear':
                    break

                o+=1
            if kernel_ == 'linear':
               break
            n +=1 
        if kernel_ == 'linear':
            break

    # if Type == 'added_connectivity':
    #     ModelName = 'add_connectivity'+trial+subtask
    #     config2={
    #         'c': best_model_[ModelName+'c'],
    #         'gamma':best_model_[ModelName+'gamma'],
    #         'kernel_': best_model_[ModelName+'kernel'],
    #         'acc_test':best_model_[ModelName+'test_acc'],
    #         'figType': 'C_added'
    #     }
    #     plot_3d_mesh(para_model_[ModelName], str(ModelName),m,config2, connections)
    # else:
    #     ModelName = 'all_connectivity'+trial+subtask
    #     m = 'None'
    #     config2={
    #         'c': best_model_[ModelName+'c'],
    #         'gamma':best_model_[ModelName+'gamma'],
    #         'kernel_': best_model_[ModelName+'kernel'],
    #         'acc_test':best_model_[ModelName+'test_acc'],
    #         'figType': 'C_all'
    #     }
    # plot_3d_mesh(para_model_[str(ModelName)], str(ModelName),m,config2, connections)
    #plot_3d_mesh(para_model_T2_, connections,'second trial')
    #plot_3d_mesh(para_model_T3_, connections,'third trial')
    return para_model_,best_model_, para_model_with_mcc, para_model_with_bas

    