import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
import pickle
from textwrap import wrap
from sklearn.cluster import KMeans
from sklearn.metrics import auc, confusion_matrix, RocCurveDisplay, classification_report, \
    accuracy_score, ConfusionMatrixDisplay, matthews_corrcoef, balanced_accuracy_score, \
    precision_score, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold
from matplotlib.pyplot import figure
# dir = os.path.dirname(__file__)

def plot_3d_mesh(z,ModelName,m,config2,best_model_,savePath_results, conn_num, connections = '', type = ''):
    model_para_grid = config2['model_para_grid']
    x = model_para_grid['c']
    y =  model_para_grid['gamma']
    kernel = best_model_[ModelName+'kernel']
    c = best_model_[ModelName+'c']
    gamma = best_model_[ModelName+'gamma']
    #print(f'kernel: {kernel}, c : {x}, gamma: {y}')
    figure(figsize=(10, 8), dpi=80)
    ax =sns.heatmap(z, annot=True, cmap='BrBG') #cmap='BrBG'
    ax.set_xticklabels(x, rotation=45,horizontalalignment='right')
    ax.set_yticklabels(y, rotation=45)
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
    #plt.show()
    plt.close()

def plot_CF_matrix(y_test_cv, y_test_pred_cv, svclassifier,connectivity_set, ModelName, set_number,savePath_results, labels, type):
    #print('y_test_cv, y_test_pred_cv', y_test_cv, y_test_pred_cv)
    #ConfusionMatrixDisplay.from_estimator(svclassifier, x_test, y_test) #, labels= ['MA','MA_reten']
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
    ttl = type+ str(ModelName)
    plt.title('\n'.join(wrap(ttl,60)))
    plt.savefig(os.path.join(savePath_results,figName))
    plt.close()

def analysis_tuned_model( data,n_folds, set_number,connectivity_set, \
    z,ModelName,m, config,best_model_,savePath_results, conn_num, connections = '', type = ''):   # analysis on model tuned with parameter search. 

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
    APclassifier = KMeans(n_clusters=2)
    cv = StratifiedKFold(n_splits=n_folds)
    acc_test_CV = []
    mcc_test_CV = []
    bas_test_CV = []
    prec_test_CV = []
    recl_test_CV = []
    spci_test_CV = []
    f1_test_CV = []
    spci_test_CV = []

    cv = StratifiedKFold(n_splits=n_folds)
    y_test_cv = []
    y_test_pred_cv = []
    X = data.iloc[:,:-1]
    # print('X.shape: ',X.shape)
    Y = data.iloc[:,-1]
    mean = X.mean(axis = 0)
    std = X.std( axis = 0)
    X = (X-mean)/std
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
        #print('Confusion Mat: ', y_test,y_pred_test)
        #print(classification_report(y_test,y_pred_test))
        #plot_confusion_matrix(svclassifier, X_test, y_test)  
        acc_test  = accuracy_score(y_test,y_pred_test)
        mcc_test  = matthews_corrcoef(y_test,y_pred_test)
        bas_test  = balanced_accuracy_score(y_test,y_pred_test)
        prec_test = precision_score(y_test,y_pred_test)
        recl_test = recall_score(y_test,y_pred_test)  # sensitivity is equall to recall
        f1_test   = f1_score(y_test,y_pred_test)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test).ravel()
        specificity = tn / (tn+fp)

        acc_test_CV.append(acc_test)
        mcc_test_CV.append(mcc_test)
        bas_test_CV.append(bas_test)
        prec_test_CV.append(prec_test)
        recl_test_CV.append(recl_test)
        spci_test_CV.append(specificity)
        f1_test_CV.append(f1_test)

        y_test_pred_cv.append(y_pred_test)
        y_test_cv.append(y_test)
    y_test_cv = np.hstack(y_test_cv)
    y_test_pred_cv = np.hstack(y_test_pred_cv)
    acc_mean = np.mean(acc_test_CV) # Mean of all the fold cross validation
    mcc_mean = np.mean(mcc_test_CV)
    bas_mean = np.mean(bas_test_CV)
    prec_mean = np.mean(prec_test_CV)
    recl_mean = np.mean(recl_test_CV)
    f1_mean = np.mean(f1_test_CV)
    spci_mean = np.mean(spci_test_CV)

    df_metrics_write = pd.DataFrame({'Kernal':kernel_,'C':c,'gamma':gamma_,'para_model_mcc': best_model_[ModelName+'test_mcc_CV'],\
                'para_model_bas': best_model_[ModelName+'test_bas_CV'],\
                'acc_mean':acc_mean,'mcc_mean':mcc_mean,'bas_mean':bas_mean,'prec_mean':prec_mean,\
                'recl_mean':recl_mean,'f1_mean':f1_mean,'spci_mean':spci_mean}, index=[0])

    #print('days, trials:', config['Days'], config['trials'])
    fname_mtx = 'Metrics'+'_set_'+str(set_number)+'_'+str(ModelName)+'.xlsx'  # storcs best model metrices.
    xls_fileName_metrics = os.path.join(savePath_results,fname_mtx)
    df_metrics_write.to_excel(xls_fileName_metrics,sheet_name='set_'+str(set_number))    
    with open(os.path.join(savePath_results,'set_'+str(set_number)+str(ModelName)+'tuned_model_variables.sav'), 'wb') as f:
            pickle.dump(best_model_, f)                     
    # with pd.ExcelWriter(xls_fileName_metrics, mode='a') as writer:          # Record the connection sets that increase the accuracy
    #     df_metrics_write.to_excel(writer,sheet_name='set_'+str(set_number))

    #labels = ['Unsuccessful','Successful']
    labels= ['Novice','Expert']
    plot_CF_matrix(y_test_cv, y_test_pred_cv, APclassifier,connectivity_set, ModelName, set_number,savePath_results, labels, type)
    #plot_ROC(APclassifier, X,Y, n_folds,set_number,savePath_results, type)

def plot_ROC(svclassifier, X,y, n_folds,set_number,savePath_results, type): 
    cv = StratifiedKFold(n_splits=n_folds)
    #target_names = 'Mqn'
    target_names = 'successful'
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    fig, ax = plt.subplots(figsize=(6, 6))
    for fold, (train_indx, test_indx) in enumerate(cv.split(X, y)):
        x_train,x_test = X.iloc[train_indx],X.iloc[test_indx]
        y_train,y_test = y.iloc[train_indx],y.iloc[test_indx]
        svclassifier.fit(x_train, y_train)
        viz = RocCurveDisplay.from_estimator(
            svclassifier,
            x_test,
            y_test,
            name=f"ROC fold {fold}",
            alpha=0.3,
            lw=1,
            ax=ax,
        )
        y_pred_test = svclassifier.predict(x_test)
        for i in range(len(y_pred_test)):
            if y_pred_test[i] ==0:
                y_pred_test[i] = -1
        # ConfusionMatrixDisplay.from_estimator(svclassifier, x_test, y_test, labels= ['SA','MA'])
        #print('Confusion Mat: ', confusion_matrix(y_test,y_pred_test))
        #print(classification_report(y_test,y_pred_test))
        #plot_confusion_matrix(svclassifier, X_test, y_test)  
        acc_test = accuracy_score(y_test,y_pred_test)
        mcc_test = matthews_corrcoef(y_test,y_pred_test)
        bas_test = balanced_accuracy_score(y_test,y_pred_test)
        
        #print(f'acc_test: {acc_test}, mcc_test: {mcc_test}, bas_test: {bas_test}')

        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)
    ax.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title=f"Mean ROC curve with variability\n(Positive label '{target_names}'): {svclassifier}",
    )
    ax.axis("square")
    ax.legend(loc="lower right")
    figName = 'ROC'+str(type)+'_set_'+str(set_number)
    plt.savefig(os.path.join(savePath_results,figName))
    #plt.show()
    plt.close()
