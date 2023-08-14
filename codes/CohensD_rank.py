from email.charset import SHORTEST
from cv2 import rotate
from matplotlib.pyplot import annotate
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import seaborn as sns
import string
from pyparsing import col
from sympy import rotations
import h5py

PATH = r'C:\Users\_Kamat_\Desktop\RPI\ResearchWork\Papers_\Effective_Connectivity\EEG_fNIRS_paper_Brain_informatics\EEG_codes_results\GC_Results_Jun10'
fileName = os.path.join(PATH,'Effect_size_Nov_Exp_alpha_Jun_10.xlsx')
sheets = ['Trial-1','Trial-2','Trial-3']
df1 = []
for sheet_ in sheets:  # loop over trial 
    df = pd.read_excel(fileName, sheet_name= sheet_, header =0, index_col=0 )
    df = df.abs()
    df = df.rank(axis =0, ascending=False, method='min')
    ax = sns.heatmap(df, annot=True,cmap='BrBG' )
    ax.set_xticklabels(ax.get_xticklabels(), rotation= 45,horizontalalignment='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation= 45)
    ttl = string.capwords("Cohen's D ")
    plt.title(ttl+'effect size:'+ str(sheet_))
    plt.show()
    df1.append(df)
    df.to_hdf('./store.h5', sheet_)  
for sheet_ in sheets:
    reread = pd.read_hdf('./store.h5', sheet_)
    print(reread)





# hf = h5py.File('data_test1.h5', 'w')
# hf.create_dataset('dataset_1', data=df1)

# df = pd.DataFrame(np.array(h5py.File('data_test1.h5')['dataset_1']))

# with h5py.File("rank_cohenD.h5", "w") as hf:
#     hf.create_dataset("arr", data=df1)