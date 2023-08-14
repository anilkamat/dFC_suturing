from ast import Mod
import numpy as np
import matplotlib.pyplot as plt
from EEG_plot_mesh_grid_08152022 import plot_3d_mesh
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, plot_confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut
import pandas as pd
import pickle
import os

def grid_search(data,connections,fnames,config,Type = 'dropped_connectivity'):
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
    
    model_para_grid = {
    'c' :np.arange(0.1,20,0.5), #[0.001, 0.01, 0.1, 1, 10, 100,1000,10000],
    'gamma':np.arange(0.1,20,0.5), #[0.001, 0.01, 0.1,1,10,100,1000],
    'kernel' : ['linear','rbf']
    }
    # model_para_grid = {
    # 'c' :[0.1],
    # 'gamma':[1],
    # 'kernel' : ['linear','rbf']
    # }
    para_model_ = {} 
    best_model_ = {}
    test_acc_dropped_ = np.zeros((20))
    column_ = ['Zero']
    trials = config['trials']
    phases= config['phases']#[0:1]
    #connections = connections#[0:2]
    # print('fnames: ',fnames)
    #print('connections: ',len(connections))
    # print('trials: ',trials)
    # print('phases: ',phases)
    for trial in trials: 
        for ST in phases: 
                if Type == 'all_connectivity':
                    ModelName = 'all_connectivity'+trial+ST
                    para_model_[ModelName]=np.zeros(((len(model_para_grid['gamma']), len(model_para_grid['c']))))
                    best_model_[ModelName+'c'] = 0.0
                    best_model_[ModelName+'gamma'] = 0.0
                    best_model_[ModelName+'kernel'] = ''
                    best_model_[ModelName+'test_acc'] = 0.0
                    best_model_[ModelName+'test_acc_CV'] =0.0
                else:
                    for m in range(20):  # for feature/connectivity dropped datasets.
                        ModelName = trial+ST+'_'+str(m)
                        print('modelname:',ModelName)
                        para_model_[ModelName]=np.zeros(((len(model_para_grid['gamma']), len(model_para_grid['c']))))
                        best_model_[ModelName+'c'] = 0.0
                        best_model_[ModelName+'gamma'] = 0.0
                        best_model_[ModelName+'kernel'] = ''
                        best_model_[ModelName+'test_acc'] = 0.0
                        best_model_[ModelName+'test_acc_CV'] =0.0
    # all_para_model_T3 ={}
    # #para_model_T1_ = np.zeros((20,7,8))   # (connection, gamma, c)
    # para_model_T2_ = np.zeros((20,7,8))   # (connection, gamma, c)
    # para_model_T3_ = np.zeros((20,7,8))   # (connection, gamma, c)
    random_states= [1]
    for random_state_ in random_states:
        for kernel_ in model_para_grid['kernel']:
            n = 0
            for gamma_ in model_para_grid['gamma']:
                o = 0
                for c_ in model_para_grid['c']:
                    if kernel_ == 'linear':
                        svclassifier = SVC(kernel=kernel_, random_state=random_state_)#random_state=1,) # ,
                    else:
                        svclassifier = SVC(kernel=kernel_, gamma=gamma_, C=c_, random_state=random_state_)# gamma=100, C=0.1,random_state=1,) # ,
                    test_acc = []
                    temp_ = np.zeros((20))
                    for ST in phases: 
                        for trial in trials: # hold out one trial (all phases) and train on remainings
                            Train_data = []
                            Test_data = []  # this testing dataset is different than the validation(CV) dataset
                            for names in fnames:  # loop over phases to extract data for a particular subtas (ST)
                                if names[-2:] == ST:
                                    df1 = data[names]  # data of each phases
                                    if names[-6:-2] == trial:
                                        Test_data.append(df1)  # Not used in CV, just separating from train_data.
                                    else:
                                        Train_data.append(df1)
                            Train_data = pd.concat(Train_data)
                            Test_data = pd.concat(Test_data)
                                # test_acc ={}
                            if Type=='all_connectivity':
                                X_train = Train_data.iloc[:,:-1]
                                Y_train = Train_data.iloc[:,-1]
                                mean = X_train.mean(axis = 0)
                                std = X_train.std( axis = 0)
                                X_train = (X_train-mean)/std
                                #print('normalized Xtrain :', X_train)

                                loo = LeaveOneOut()
                                loo.get_n_splits(X_train)
                                acc_test_CV = []
                                for train_indx, test_indx in loo.split(X_train):
                                    x_train,x_test = X_train.iloc[train_indx],X_train.iloc[test_indx]
                                    y_train,y_test = Y_train.iloc[train_indx],Y_train.iloc[test_indx]
                                    acc_subtask_train = SVM(x_train,y_train)
                                    acc_subtask_test = SVM_test(x_test,y_test)
                                    acc_test_CV.append(acc_subtask_test)
                                acc_test_CV_mean = np.mean(acc_test_CV)
                                acc_test_CV_std = np.std(acc_test_CV)
                                #print(f'Train acc:{acc_train}')
                                accuracies = []
                                # test_acc[connection] = []
                                X_test = Test_data.iloc[:,:-1]
                                y_test = Test_data.iloc[:,-1]
                                X_test = (X_test-mean)/std

                                acc_test = SVM_test(X_test,y_test)
                                accuracies.append(acc_test)
                                ModelName = 'all_connectivity'+trial+ST
                                print('model_name:',ModelName)
                                #para_model_T1_[ModelName[:-4]] = np.zeros((20,7,8))   # (connection, gamma, c)
                                if (acc_test_CV_mean > best_model_[ModelName+'test_acc_CV']) :
                                    #save the model
                                    pickle.dump(svclassifier, open(ModelName+'.sav', 'wb'))
                                    best_acc_test1 = acc_test
                                    best_model_[ModelName+'c'] = c_
                                    best_model_[ModelName+'gamma'] = gamma_
                                    best_model_[ModelName+'kernel']= kernel_
                                    best_model_[ModelName+'test_acc'] = acc_test
                                    best_model_[ModelName+'test_acc_CV'] =acc_test_CV_mean
                                # store all the parameters and corrosponding accuracies
                                para_model_[ModelName][n,o] = acc_test_CV_mean
                                test_acc.append(best_model_[ModelName+'test_acc'])
                                #print('para_model_T1: ',para_model_[ModelName])
                                
                            elif Type=='dropped_connectivity':
                                m = 0 # stores acc for each dropped connectivity
                                accuracies = []
                                for connection in connections:
                                    Train_data_dropped = Train_data.drop(connection, axis = 1, inplace=False)
                                    X_train = Train_data_dropped.iloc[:,:-1]
                                    Y_train = Train_data_dropped.iloc[:,-1]
                                    #print('dropped connec:',connection)
                                    #print('X_train:',X_train)
                                    #print('Y_train:',Y_train)
                                    mean = X_train.mean(axis = 0)
                                    std = X_train.std( axis = 0)
                                    X_train = (X_train-mean)/std
                                    #print('mean,std :', mean,std)
                                    loo = LeaveOneOut()
                                    loo.get_n_splits(X_train)
                                    acc_test_CV = []
                                    for train_indx, test_indx in loo.split(X_train):
                                        # print(train_indx)
                                        # print(test_indx)
                                        # print(X_train)
                                        x_train,x_test = X_train.iloc[train_indx],X_train.iloc[test_indx]
                                        y_train,y_test = Y_train.iloc[train_indx],Y_train.iloc[test_indx]
                                        acc_subtask_train = SVM(x_train,y_train)
                                        acc_subtask_test = SVM_test(x_test,y_test)
                                        acc_test_CV.append(acc_subtask_test)
                                    acc_test_CV_mean = np.mean(acc_test_CV)
                                    acc_test_CV_std = np.std(acc_test_CV)
                                    #print(f'Test CV Mean_acc:{acc_test_CV_mean}')
                                    #test_acc[connection] = []
                                    Test_data_dropped =  Test_data.drop(connection, axis = 1, inplace=False)
                                    ModelName = trial+ST+'_'+str(m)
                                    print('Running:',ModelName[:-2])
                                    #print('test_data_dropped: ',Test_data_dropped)
                                    X_test = Test_data_dropped.iloc[:,:-1]
                                    y_test = Test_data_dropped.iloc[:,-1]
                                    X_test = (X_test-mean)/std
                                    #print('test_data_dropped: ',X_test)
                                    acc_test = 0.0
                                    acc_test = SVM_test(X_test,y_test)
                                    accuracies.append(acc_test)
                                    #print('acc_test: ',acc_test)
                                    #print('accuracices: ',accuracies)
                                    #print('best acc for the connectivity+model: ',best_model_[ModelName+'test_acc'])
                                    if (acc_test_CV_mean > best_model_[ModelName+'test_acc_CV']) :
                                        #save the model
                                        #pickle.dump(svclassifier, open(ModelName+'.sav', 'wb')) 
                                        best_acc_test1 = acc_test
                                        best_model_[ModelName+'c'] = c_
                                        best_model_[ModelName+'gamma'] = gamma_
                                        best_model_[ModelName+'kernel']= kernel_
                                        best_model_[ModelName+'test_acc'] = acc_test
                                        #print('model best acc: ',acc_test)
                                        best_model_[ModelName+'test_acc_CV'] =acc_test_CV_mean
                                        best_model_[ModelName+'dropped_connection'] = connection
                                    # store all the parameters and corrosponding accuracies
                                    #print('model Name: ',ModelName)
                                    para_model_[ModelName][n,o] = acc_test_CV_mean
                                    test_acc_dropped_[m] = best_model_[ModelName+'test_acc']  # store the results form the best models
                                    #print('para_model_T1: ',para_model_[ModelName])
                                    m += 1
                                temp_ = np.vstack((temp_,test_acc_dropped_))
                                #column_.append(trial+ST+'.csv')
                                # df3 = pd.DataFrame(test_acc_dropped_)
                                # FullFilename2 = os.path.join(config['PATH2'],column_)
                                # df3.to_csv(FullFilename2)
                        if Type == 'all_connectivity':
                            df3 = pd.DataFrame(test_acc)
                            df3 = df3.T
                            FullFilename2 = os.path.join(config['PATH2'],'EEG_allTrials'+'Phases_RanSt'+str(random_state_)+'.csv')
                            df3.to_csv(FullFilename2)
                    if Type == 'dropped_connectivity':
                        #print('temp_ : size',temp_.shape)
                        df3 = pd.DataFrame(temp_)
                        df3 = df3.T
                        FullFilename2 = os.path.join(config['PATH2'],'EEG_dropped_connectivity_Phases_Trials'+'.csv')
                        # print(column_)
                        df3.to_csv(FullFilename2) #,header=column_
                    o+=1
                n +=1 
        for trial in trials:
            for ST in phases:
                if Type == 'dropped_connectivity':
                    for m in range(12):
                        ModelName = trial+ST+'_'+str(m)
                        config2={
                            'c': best_model_[ModelName+'c'],
                            'gamma':best_model_[ModelName+'gamma'],
                            'kernel_': best_model_[ModelName+'kernel'],
                            'acc_test':best_model_[ModelName+'test_acc'],
                            'figType': 'C_dropped'
                        }
                        plot_3d_mesh(para_model_[ModelName], str(ModelName),m,config2, connections)
                else:
                    ModelName = 'all_connectivity'+trial+ST
                    m = 'None'
                    config2={
                        'c': best_model_[ModelName+'c'],
                        'gamma':best_model_[ModelName+'gamma'],
                        'kernel_': best_model_[ModelName+'kernel'],
                        'acc_test':best_model_[ModelName+'test_acc'],
                        'figType': 'C_all'
                    }
                    plot_3d_mesh(para_model_[str(ModelName)], str(ModelName),m,config2, connections)
    #plot_3d_mesh(para_model_T2_, connections,'second trial')
    #plot_3d_mesh(para_model_T3_, connections,'third trial')
    # baseName = '/content/gdrive/MyDrive/data_used_colab/EEG_fNIRS_suturing/'
    return para_model_,best_model_

    