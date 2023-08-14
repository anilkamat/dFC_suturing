import numpy as np
import pandas as pd
import seaborn as sns; sns.set_theme()
import matplotlib.pyplot as plt
import os

def rankVariable(file,sheets):
    df_ranks = []
    for sheet_ in sheets:
        df1 = pd.read_excel(file,sheet_name=sheet_, header=0,index_col = 0)
        df1 = df1.rank( axis =0,ascending=True, method = 'min') #df1.rank(method='max')
        ax =sns.heatmap(df1, annot=True,cmap='BrBG')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45,horizontalalignment='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=45)
        plt.show()
        df_ranks.append(df1)
    return df_ranks

