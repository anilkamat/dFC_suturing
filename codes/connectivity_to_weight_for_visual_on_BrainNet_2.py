from email import header
import pandas as pd
import numpy as np
import os


""" The code has been used to find the contribution of each connectivity to the accuracy of kSVM/1DCNN model
.It converts the cummulative accuracy into individual contribution and writes them to the csv and .edge file. 
.edge file is used in BrainNet for visualization of connectivity onto brain. Just the file path can be changed for
different experiment/comparision"""

config = {
    #'phase':['P1','P2','P3','P4'],#'D3'],#'D3'],
    'phase':['ST2','ST4','ST5','ST6','ST7','ST8','ST9','ST10','ST11','ST12','ST13'],#'D3'],#'D3'],
}
connection_GC = [[' ','L PFC-->RPFC','LPFC-->LM1','LPFC-->RM1','LPFC-->SMA'], 
    ['RPFC-->LPFC',' ','RPFC-->LM1','RPFC-->RM1','RPFC-->SMA'],
    ['LM1-->LPFC','LM1-->RPFC',' ','LM1-->RM1','LM1-->SMA'],
    ['RM1-->LPFC','RM1-->RPFC','RM1-->LPMC',' ','RM1-->SMA'],
    ['SMA-->LPFC','SMA-->RPFC','SMA-->LM1','SMA-->RM1',' ']]
connection_GC = connection_GC
connection_GC = list(map(list, zip(*connection_GC)))
# import .txt file 
filename = 'top_connectivity_and_cumulative_accuracies.txt'
savefilename = 'connectivity_matrix_for_BrainNetVisual.edge'
csvsavefilename = 'connectivity_matrix_for_BrainNetVisual.csv'
# cumulative accuracy of each of the connectivity calculated from kSVM with feature selection.
num_region = 5
for phase in config['phase']:

    print(phase+str('_')+filename)
    connectivity_acc = pd.read_csv(os.path.join(r'C:\Users\_Kamat_\Desktop\RPI\ResearchWork\Papers_\Effective_Connectivity\EEG_fNIRS_paper_Brain_informatics\channelEEG_codes_results_alphaBand\Results\Connectivities_LSTMED_multiChanROI_Copy',phase+str('_')+filename), sep=' ', header=None)
    savepath = os.path.join(r'C:\Users\_Kamat_\Desktop\RPI\ResearchWork\Papers_\Effective_Connectivity\EEG_fNIRS_paper_Brain_informatics\channelEEG_codes_results_alphaBand\Results\Connectivities_LSTMED_multiChanROI_Copy',phase+savefilename)
    savepath2 = os.path.join(r'C:\Users\_Kamat_\Desktop\RPI\ResearchWork\Papers_\Effective_Connectivity\EEG_fNIRS_paper_Brain_informatics\channelEEG_codes_results_alphaBand\Results\Connectivities_LSTMED_multiChanROI_Copy',phase+csvsavefilename)


    num_conn = connectivity_acc.shape[0]
    # for i in range(num_conn-1,0,-1): 
    #     connectivity_acc.iloc[i,1] = connectivity_acc.iloc[i,1]-connectivity_acc.iloc[i-1,1]
    temp_mat = np.zeros((num_region,num_region))
    for i in range(num_conn):
        for j in range(num_region):
            for k in range(num_region):
                # print(connectivity_acc.iloc[i,0])
                # print(connection_GC[j][k])
                if connectivity_acc.iloc[i,0] == connection_GC[j][k]:
                    temp_mat[j,k] = connectivity_acc.iloc[i,1]
                    #print('temp_mat: ',connection_GC[j][k], temp_mat)
    #print('temp_mat: ',temp_mat)
    temp_mat =  pd.DataFrame(temp_mat)
    temp_mat.to_csv(savepath, header=None,sep = ' ', index=None)
    #temp_mat =  pd.DataFrame(temp_mat)
    #connectivity_acc.to_csv(savepath2, header=None,sep = ' ', index=None)
    #write to a txt file with .edge extension.




