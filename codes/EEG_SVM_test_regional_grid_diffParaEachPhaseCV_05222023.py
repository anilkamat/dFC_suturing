import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import time
import os
from EEG_grid_search_parameters_subtask_08152022 import grid_search
import warnings
warnings.filterwarnings('ignore')
s_time = time.time()
config={
  'PATH' :r'C:\Users\_Kamat_\Desktop\RPI\ResearchWork\Papers_\Effective_Connectivity\EEG_fNIRS_paper_Brain_informatics\EEG_codes_results\GC_regional_Jun10\Connectivities\Phases',
  'PATH2' :r'C:\Users\_Kamat_\Desktop\RPI\ResearchWork\Papers_\Effective_Connectivity\EEG_fNIRS_paper_Brain_informatics\EEG_codes_results\GC_regional_Jun10\SVM_results',
  'phases':['P1','P2','P3','P4'],
  'trials' : ['Tr_1','Tr_2','Tr_3']
}
filenames = []
Data_all = [] # all subjects, all trial, all HTA
data = {}
fnames = []
remove_columns = ['LPFC-->SMA','RPFC-->SMA','LPMC-->SMA','RPMC-->SMA',\
                  'SMA-->LPFC','SMA-->RPFC','SMA-->LPMC','SMA-->RPMC'] # remove columns which don't have enough number of ICs (ex: SMA)
for filename in os.listdir(config['PATH']):
    if filename.endswith("csv"): 
        df = pd.read_csv(os.path.join(config['PATH'], filename), header = 0, index_col=False, sep=',')
        #df = df[(df != 0).all(1)]  # remove rows with zeros.
        df.drop(remove_columns, axis=1, inplace=True) # removes features which donot have enough ICs (i.e. SMA) region.
        df = df.loc[(df != 0).any(axis=1)]
        Data_all.append(df)
        filenames.append(filename)
        fname = filename.strip('EEG_GC_Exp_Nov')
        fname = fname.strip('.cs')
        fnames.append(fname)
        data[fname] = df
Test_data = Data_all
Data_all = pd.concat(Data_all)
connections = df.columns[0:12]

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

# print('Analysis of droppedconnectivity')
# para_model_,best_model_ =grid_search(data,connections,fnames,config,Type='dropped_connectivity')  # grid search for best parameters 
# pickle.dump(para_model_, open('para_model_dropped'+'.sav', 'wb'))
# pickle.dump(best_model_, open('best_model_dropped'+'.sav', 'wb'))
print('Analysis of fullconnectivity')
para_model_allConnectivity,best_model_allConnectivity =grid_search(data,connections,fnames,config,Type='all_connectivity')  # grid search for best parameters 
pickle.dump(para_model_allConnectivity, open('para_model_dropped_allC'+'.sav', 'wb'))
pickle.dump(best_model_allConnectivity, open('best_model_dropped_allC'+'.sav', 'wb'))
test_acc = {}
print('Execution time :',time.time()-s_time)
