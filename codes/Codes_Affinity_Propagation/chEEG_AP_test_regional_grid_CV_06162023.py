from email import header
from math import gamma
from operator import index
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
# from Rank_variables_SVM_accuracy import rankVariable
from chEEG_grid_search_parameters_subtask_06162023 import grid_search
from chEEG_plot_mesh_grid_06162023 import plot_3d_mesh, analysis_tuned_model
import warnings
warnings.filterwarnings('ignore')
s_time = time.time()
config={
  'trials':['T123'],#],'T2','T3'], 
  #'phases' : ['P1','P2','P3','P4'],   
  'phases' : ['ST1','ST2','ST3','ST4','ST5','ST6','ST7','ST8','ST9','ST10','ST11','ST12','ST13'],   
    'model_para_grid':{'c' :np.arange(0.001,1,1),  # 'model_para_grid':{'c' :np.arange(0.001,10,0.5),# if change value, change in the grid search fxn as well
    'gamma':np.arange(0.001,1,1),   #'gamma':np.arange(0.001,10,0.5), 
    'kernel_' : ['rbf']}
}
n_folds = 5 # For crossvalidation  
filenames = []
Data_all = []           # all Subjects, all trial, all HTA
for T in config['trials']:
    for P in config['phases']:
        # filename = os.path.join(config['PATH'], file_)
        #filename = r'C:\Users\kamata3\Work\Brain\Brain_network\Stressor_arm\Results\connectivities\normal_vs_stressor\fNIRS_GC_normalVsSA_D123T1_5_Succ_n_Unsucc.csv'
        csvfilename = 'EEG_GC_Exp_Nov_'+str(T)+str(P)+'.csv'
        print('working on csvfilename:', csvfilename)
        filename = os.path.join(r'C:\Users\kamata3\Work\Brain\Brain_network\EEG_fNIRS_paper_Brain_informatics\channelEEG_codes_results_alphaBand\Results\Connectivities',csvfilename)
        savePath_results =  os.path.join(r'C:\Users\kamata3\Work\Brain\Brain_network\EEG_fNIRS_paper_Brain_informatics\channelEEG_codes_results_alphaBand\Results\AP_07202023',T+P)        
        df = pd.read_csv(filename, header=0, index_col=False, sep=',')
        df = df[(df!=0).all(1)]
        connections = df.columns[0:20]
        config_trial={'filename': csvfilename,     # config of current subtasks 
        'phase':P,
        'trial':T,
        'connections': connections}

        print('############ Analysis of fullconnectivity ##############')
        print('shape of df:',df.shape)
        set_number = 0  # set_number=0 is for the baseline set .i.e set with all connectivity
        para_model_allConnectivity, best_model_allConnectivity, para_model_allConnectivity_mcc, para_model_allConnectivity_bas \
            = grid_search(df,config_trial,config,n_folds,set_number,savePath_results, Type='all_connectivity')  # grid search for best parameters 
        ModelName = 'all_connectivity_'+T+P
        acc_allC = best_model_allConnectivity[ModelName+'test_acc_CV']
        m= 'all_C'
        plot_3d_mesh(para_model_allConnectivity[str(ModelName)], str(ModelName),m,config,best_model_allConnectivity,savePath_results,connections, type='acc')
        analysis_tuned_model(df,n_folds,set_number,connections, para_model_allConnectivity[str(ModelName)], str(ModelName),m,config,best_model_allConnectivity,savePath_results,connections, type='acc')
        plot_3d_mesh(para_model_allConnectivity_mcc[str(ModelName)], str(ModelName),m,config,best_model_allConnectivity,savePath_results,connections, type = 'mcc')
        print('baseline accuracy: ',acc_allC)
        df_write = pd.DataFrame({'acc':acc_allC},index=['1']) #,index=False,header=0
        fname = T+P+'.xlsx'
        xls_fileName = os.path.join(savePath_results,fname)
        df_write.to_excel(xls_fileName,sheet_name='set_'+str(set_number))    

        print('######### Add_connectivity analysis ############')
        dict = {}
        ModelName = 'add_connectivity_'+T+P
        m = 'single_C_add'
        i = 0   # connection number
        set_number = '_dummy' # this set is just used as buffer/ dummy set_number
        for connection in connections:  # test one single connectivity at a time and select top performer for futher forming sets for refin_connectivity
            data = df[[connection,'Class']]  
            para_model_,best_model_, para_model_mcc, para_model_bas \
                =grid_search(data,config_trial,config,n_folds,set_number,savePath_results,Type='add_connectivity')  # grid search for best parameters 
            dict[connection] = best_model_[ModelName+'test_acc_CV']
            plot_3d_mesh( para_model_[str(ModelName)], str(ModelName),m,config,best_model_,savePath_results, i, connection, type = 'acc')
            plot_3d_mesh( para_model_mcc[str(ModelName)], str(ModelName),m,config,best_model_,savePath_results, i, connection, type = 'mcc')
            plot_3d_mesh( para_model_bas[str(ModelName)], str(ModelName),m,config,best_model_,savePath_results, i, connection, type = 'bas')
            analysis_tuned_model(data,n_folds,set_number,connection, para_model_[str(ModelName)], str(ModelName),m,config,best_model_,savePath_results,i, type='acc')
            i+=1
        sort_orders = sorted(dict.items(), key=lambda x: x[1], reverse=True)
        #print('sort_orders: ',sort_orders)
        selected_conn = sort_orders[0:1]
        #print('selected_conn: ',selected_conn[0][0])
        top_single_conn = selected_conn[0][0]

        set_number = 1
        for i in range(1): # for loop to write the metrices of the single_connectivity with maximum accuracy
            data = df[[top_single_conn,'Class']]  
            para_model_,best_model_, para_model_mcc, para_model_bas \
                =grid_search(data,config_trial,config,n_folds,set_number,savePath_results,Type='add_connectivity')  # grid search for best parameters 
            dict[top_single_conn] = best_model_[ModelName+'test_acc_CV']
            plot_3d_mesh( para_model_[str(ModelName)], str(ModelName),m,config,best_model_,savePath_results, i, top_single_conn, type = 'acc')
            plot_3d_mesh( para_model_mcc[str(ModelName)], str(ModelName),m,config,best_model_,savePath_results, i, top_single_conn, type = 'mcc')
            plot_3d_mesh( para_model_bas[str(ModelName)], str(ModelName),m,config,best_model_,savePath_results, i, top_single_conn, type = 'bas')
            analysis_tuned_model(data,n_folds,set_number,top_single_conn, para_model_[str(ModelName)], str(ModelName),m,config,best_model_,savePath_results,i, type='acc')

        def refine_connectivity(set1,set3,df,set1_top_acc,set_number): # refines the discering connectivities.
            dict2={}
            for connection in set3:
                set1 = set1+[connection]  # add a new connection from set3 to set1
                data = df[set1+['Class']]
                para_model_,best_model_,para_model_mcc, para_model_bas \
                    =grid_search(data,config_trial,config,n_folds,set_number,savePath_results, Type='add_connectivity')  # grid search for best parameters 
                acc_set1 = best_model_[ModelName+'test_acc_CV']
                if acc_set1 > set1_top_acc:  # if acc increased
                    #print('accuracy increased! keep searching')
                    #tuned_model(data, para_model_)   # compute the metrics for the tuned (c,gamma) model and plot them for visualization
                    set_number += 1
                    set1_top_acc = acc_set1  # increased accuracy
                    dict2[tuple(set1)] = best_model_[ModelName+'test_acc_CV']
                    # print('para_model_: ',para_model_[ModelName])
                    # df_metrics_write = pd.DataFrame({'para_model_mcc': best_model_[ModelName+'test_mcc_CV'],\
                    #           'para_model_bas': best_model_[ModelName+'test_bas_CV']}, index=[0])
                    # with pd.ExcelWriter(xls_fileName_metrics, mode='a') as writer:          # Record the connection sets that increase the accuracy
                    #     df_metrics_write.to_excel(writer,sheet_name='set_'+str(set_number))
                    
                    df_write = pd.DataFrame({'set':set1,'acc':set1_top_acc})
                    with pd.ExcelWriter(xls_fileName, mode='a') as writer:          # Record the connection sets that increase the accuracy
                        df_write.to_excel(writer,sheet_name='set_'+str(set_number))

                    plot_3d_mesh( para_model_[str(ModelName)], str(ModelName),set_number,config,best_model_,savePath_results,connections, type = 'acc')
                    plot_3d_mesh( para_model_mcc[str(ModelName)], str(ModelName),set_number,config,best_model_,savePath_results,connections, type = 'mcc')
                    plot_3d_mesh( para_model_bas[str(ModelName)], str(ModelName),set_number,config,best_model_,savePath_results,connections, type = 'bas')
                    
                    analysis_tuned_model(data,n_folds,set_number,connection, para_model_[str(ModelName)], str(ModelName),m,config,best_model_,savePath_results,i, type='acc')
                    #print('testCV_acc: ',best_model_[ModelName+'test_acc_CV'])
                    #print('connectivity set1: ',set1)
                    set3 = set3.drop(connection)
                    break
                else:
                    #print(f'accuracy decreased! for {connection} stop searching')
                    set3 = set3.drop(connection)   # dorp the connection if it didn'P imporve the accuracy
                    set1.remove(connection) # remove because it was added above to construct the dataset for grid_search. 
                    #print(f'set1.type {type(set1)}, set1: {set1}')
                    #print(f'len(set3): {len(set3)} connectivity set3: {set3}')
                    break
            print(f'set1: {set1}, set2: {set3}')
            return set1, set3,set1_top_acc,dict2,set_number

        set1 = [selected_conn[0][0]] # always selects the top performing connectivity
        set_number =1
        df_write = pd.DataFrame(selected_conn)
        with pd.ExcelWriter(xls_fileName, mode='a') as writer:  
            df_write.to_excel(writer,sheet_name='set_'+str(set_number)) # write accuracy for a single connectivity
        set3 = connections.drop(set1)  # set after dropping the set1 connetivity
        set1_top_acc = selected_conn[0][1]
        print('set#1: ',selected_conn)
        flag = 1 # continue searching if flage = 1
        dict2 = selected_conn
        while(flag):  # loop for other(except baseline and single drop) sets of connectivities
            # set2 = connections
            if len(set3) >=1:
                set1,set3,set1_top_acc,dict2,set_number= refine_connectivity(set1,set3,df,set1_top_acc,set_number)
            else: 
                flag = 0
        # df3 = pd.DataFrame(dict2)
        # df3 = df3.T
        # FullFilename2 = os.path.join(config['PATH2'],str(trial),str(ST)+'.csv')
        # df3.to_csv(FullFilename2)
        # print(f'final set1: {set1}, acc: {set1_top_acc} and set3: {set3}')
print(f'Execution time :,{(time.time()-s_time)/60} minutes')

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