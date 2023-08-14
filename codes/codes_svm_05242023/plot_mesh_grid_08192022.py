import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
import pickle
from textwrap import wrap
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import auc, confusion_matrix, RocCurveDisplay, classification_report, \
    accuracy_score, ConfusionMatrixDisplay, matthews_corrcoef, balanced_accuracy_score, \
    precision_score, f1_score, recall_score
from sklearn.model_selection import LeaveOneOut
from matplotlib.pyplot import figure
dir = os.path.dirname(__file__)

#path_ = r'C:\Users\_Kamat_\Desktop\RPI\ResearchWork\Papers_\Effective_Connectivity\EEG_fNIRS_paper_Brain_informatics\EEG_codes_results\SVM_results_05242023\T3P4'

def plot_3d_mesh(z,ModelName,m,config2,best_model_, savePath_results, connections = '', type = '', conn_num = ''):
    model_para_grid = config2['model_para_grid']
    x = model_para_grid['c']
    y =  model_para_grid['gamma']
    kernel = best_model_[ModelName+'kernel']
    c = best_model_[ModelName+'c']
    gamma = best_model_[ModelName+'gamma']
    #print(f'kernel: {kernel}, c : {x}, gamma: {y}')
    figure(figsize=(10, 8), dpi=80)

    num_elements = len(x)
    x_Tick_List = []
    for item in range (0,num_elements):
        x_Tick_List.append(x[item])
    
    num_elements = len(y)
    y_Tick_List = []
    for item in range (0,num_elements):
        y_Tick_List.append(y[item])

    ax =sns.heatmap(z, annot=True, cmap='BrBG') #cmap='BrBG'
    ax.set_xticklabels(x_Tick_List, rotation=45,horizontalalignment='right')
    ax.set_yticklabels(y_Tick_List, rotation=45)
    plt.xlabel('C')
    plt.ylabel('gamma')
    if m == 'all_C':
        ttl= str(type)+ModelName,',c',c,',gamma',gamma,',kernel',kernel #,',best_test_acc',str('{:.3f}'.format(config2['acc_test'])
        figName = str(type)+ModelName+'all_C'
    elif m =='single_C_add':
        ttl= str(type)+ModelName,'C_add_',str(connections),',c',c,',gamma',gamma,',kernel',kernel    #,',best_test_acc',str('{:.3f}'.format(config2['acc_test'])
        figName = str(type)+ModelName+'conn#'+str(conn_num)
    else:
        ttl= str(type)+ModelName,'set#',m,',c',c,',gamma',gamma,',kernel',kernel  #,',best_test_acc',str('{:.3f}'.format(config2['acc_test'])
        figName = str(type)+ModelName+'set#'+str(m)
    plt.title(ttl)
    plt.savefig(os.path.join(savePath_results,figName))
    #plt.savefig(os.path.join(dir,figName))
    #plt.show()
    plt.close()

def plot_CF_matrix(y_test_cv, y_test_pred_cv, svclassifier,connectivity_set, ModelName, set_number,savePath_results, labels, type):
    #print('y_test_cv, y_test_pred_cv', y_test_cv, y_test_pred_cv)
    #ConfusionMatrixDisplay.from_estimator(svclassifier, x_test, y_test) #, labels= ['SA','MA']
    cm=confusion_matrix(y_test_cv, y_test_pred_cv)
    # tn, fp, fn, tp=confusion_matrix(y_test_cv, y_test_pred_cv)
    # specificity = tn / (tn+fp)
    report = classification_report(y_test_cv, y_test_pred_cv, output_dict=True)
    report = pd.DataFrame(report).transpose()
    fileName = 'CF_'+str(type)+str(ModelName)+'_'+str(set_number)+'.xlsx'# +'C_'+str(svclassifier.C)+' gamma_'+str(svclassifier.gamma)
    xls_fileName = os.path.join(savePath_results, fileName)
    report.to_excel(xls_fileName)
    disp= ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=labels)
    disp.plot()
    figName = 'CF_'+str(type)+str(ModelName)+'_'+str(set_number)# +'C_'+str(svclassifier.C)+' gamma_'+str(svclassifier.gamma)
    #print('figName :', figName)
    #print('svclassifier: ', svclassifier, svclassifier.C)
    ttl = type+' C:'+str(svclassifier.C)+' gamma:'+str(svclassifier.gamma)+ str(ModelName)
    plt.title('\n'.join(wrap(ttl,60)))
    plt.savefig(os.path.join(savePath_results,figName))
    plt.close()

def analysis_tuned_model( data, set_number,connectivity_set, \
    z,ModelName,m, config,best_model_,savePath_results, connections = '', type = '',  conn_num=''):   # analysis on model tuned with parameter search. 

    model_para_grid = config['model_para_grid'] 
    x = model_para_grid['c']
    y =  model_para_grid['gamma']
    kernel_ = best_model_[ModelName+'kernel']
    c = best_model_[ModelName+'c']
    gamma_ = best_model_[ModelName+'gamma']
    #print(f'kernel: {kernel}, c : {x}, gamma: {y}')
    figure(figsize=(10, 8), dpi=80)
    ax =sns.heatmap(z, annot=True,cmap='BrBG')#cmap='BrBG'
    ax.set_xticklabels(x, rotation=45,horizontalalignment='right')
    ax.set_yticklabels(y, rotation=45)
    plt.xlabel('C')
    plt.ylabel('gamma')
    if m == 'all_C':
        ttl= str(type)+ModelName,',c',c,',gamma',gamma_,',kernel',kernel_ #,',best_test_acc',str('{:.3f}'.format(config2['acc_test'])
        figName = str(type)+ModelName+'all_C'
    elif m =='single_C_add':
        ttl= str(type)+ModelName,'C_add_',str(connections),',c',c,',gamma',gamma_,',kernel',kernel_    #,',best_test_acc',str('{:.3f}'.format(config2['acc_test'])
        figName = str(type)+ModelName+'conn#'+str(conn_num)
    else:
        ttl= str(type)+ModelName,'set#',m,',c',c,',gamma',gamma_,',kernel',kernel_  #,',best_test_acc',str('{:.3f}'.format(config2['acc_test'])
        figName = str(type)+ModelName+'set#'+str(m)
    plt.title(ttl)
    plt.savefig(os.path.join(savePath_results,figName))
    #plt.savefig(os.path.join(dir,figName))
    #plt.show()
    plt.close()
    # print('optimal kernel, gamma, c', kernel_, gamma_, c)
    svclassifier = SVC(kernel=kernel_, gamma=gamma_, C=c, random_state=1)
    acc_test_cv = []
    mcc_test_cv = []
    bas_test_cv = []
    prec_test_cv = []
    recl_test_cv = []
    specificity_test_cv = []
    f1_test_cv = []
    y_test_cv = []
    y_test_pred_cv = []

    X = data.iloc[:,:-1]
    Y = data.iloc[:,-1]
    mean = X.mean(axis = 0)
    std = X.std( axis = 0)
    X = (X-mean)/std
    loo = LeaveOneOut()
    loo.get_n_splits(X)

    for train_indx, test_indx in loo.split(X):
        x_train,x_test = X.iloc[train_indx],X.iloc[test_indx]
        y_train,y_test = Y.iloc[train_indx],Y.iloc[test_indx]
        #acc_train = SVM(x_train,y_train, svclassifier)
        svclassifier.fit(x_train, y_train)
        
        y_pred_test = svclassifier.predict(x_test)

        y_test_pred_cv.append(y_pred_test)
        y_test_cv.append(y_test)
    y_test_cv = np.hstack(y_test_cv)
    y_test_pred_cv = np.hstack(y_test_pred_cv)
    
    tn, fp, fn, tp = confusion_matrix(y_test_cv, y_test_pred_cv,  labels=[-1, 1]).ravel()
    specificity_test_cv = tn / (tn+fp)
    acc_test_cv  = accuracy_score(y_test_cv,y_test_pred_cv)
    mcc_test_cv  = matthews_corrcoef(y_test_cv,y_test_pred_cv)
    bas_test_cv  = balanced_accuracy_score(y_test_cv,y_test_pred_cv)
    prec_test_cv = precision_score(y_test_cv,y_test_pred_cv)
    recl_test_cv = recall_score(y_test_cv,y_test_pred_cv)  # sensitivity is equall to recall
    f1_test_cv   = f1_score(y_test_cv,y_test_pred_cv)

    df_metrics_write = pd.DataFrame({'Kernal':kernel_,'C':c,'gamma':gamma_,'para_model_mcc': best_model_[ModelName+'test_mcc_CV'],\
                'para_model_bas': best_model_[ModelName+'test_bas_CV'],\
                'acc_test_cv':acc_test_cv,'mcc_test_cv':mcc_test_cv,'bas_test_cv':bas_test_cv,'prec_test_cv':prec_test_cv,\
                'recl_test_cv':recl_test_cv,'f1_test_cv':f1_test_cv,'spci_test_cv':specificity_test_cv}, index=[0])

    #print('days, trials:', config['Days'], config['trials'])
    fname_mtx = 'Metrics'+'_set_'+str(set_number)+str(ModelName)+'.xlsx'  # storcs best model metrices.
    xls_fileName_metrics = os.path.join(savePath_results,fname_mtx)
    df_metrics_write.to_excel(xls_fileName_metrics,sheet_name='set_'+str(set_number))     
    with open(os.path.join(savePath_results,'set_'+str(set_number)+str(ModelName)+'tuned_model_variables.sav'), 'wb') as f:
            pickle.dump(best_model_, f)             
    # with pd.ExcelWriter(xls_fileName_metrics, mode='a') as writer:          # Record the connection sets that increase the accuracy
    #     df_metrics_write.to_excel(writer,sheet_name='set_'+str(set_number))
    labels = ['Novice', 'Expert']
    plot_CF_matrix(y_test_cv, y_test_pred_cv, svclassifier,connectivity_set, ModelName, set_number, savePath_results, labels, type)
    plot_ROC(svclassifier, y_test_cv,y_test_pred_cv,set_number,savePath_results, type)


def plot_ROC(svclassifier, y_test, y_test_pred,set_number, savePath_results, type):
    fpr, tpr, _ = metrics.roc_curve(y_test, y_test_pred)
    roc_auc = metrics.auc(fpr, tpr)
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
    
    display.plot()
    plt.plot([0,1],[0,1], linestyle='dashed', color='gray')
    plt.title('Suturing_'+str(svclassifier))
    figName = 'ROC'+str(type)+str(set_number)
    plt.savefig(os.path.join(savePath_results,figName))
    #plt.show()
    plt.close()
