import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.pyplot import figure


def plot_3d_mesh(z,ModelName,m,config2, connections):
    x= np.arange(0.1,20,0.5)
    y = np.arange(0.1,20,0.5)
    figure(figsize=(10, 8), dpi=80)
    ax =sns.heatmap(z, annot=True,cmap='BrBG')#cmap='BrBG'
    ax.set_xticklabels(x, rotation=45,horizontalalignment='right')
    ax.set_yticklabels(y, rotation=45)
    plt.xlabel('C')
    plt.ylabel('gamma')
    if m == 'None':
        ttl= ModelName,',c',str(config2['c']),',gamma',str(config2['gamma']),',kernel',str(config2['kernel_']),',best_test_acc',str('{:.3f}'.format(config2['acc_test']))
    else:
        ttl= ModelName,connections[m],',c',str(config2['c']),',gamma',str(config2['gamma']),',kernel',str(config2['kernel_']),',best_test_acc',str('{:.3f}'.format(config2['acc_test']))
    plt.title(ttl)
    figName = config2['figType']+ModelName
    plt.savefig(os.path.join(r'C:\Users\_Kamat_\Desktop\RPI\ResearchWork\Papers_\Effective_Connectivity\EEG_fNIRS_paper_Brain_informatics\EEG_codes_results\GC_regional_Jun10\SVM_results\EEG_figures_05222023',figName))
    #plt.show()
    plt.close()