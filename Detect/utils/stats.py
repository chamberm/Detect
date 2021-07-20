from __future__ import division, print_function, absolute_import
from scipy.ndimage.filters import gaussian_filter1d
from scipy import stats

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import seaborn as sns

from matplotlib.backends.backend_agg import RendererAgg
_lock = RendererAgg.lock

def correlate(method, df_demog, df_data, y_axis, savename, filename):
    #filename manip

    df_plot = pd.concat([df_data['Dist'], df_data['Group'], df_demog[y_axis]], axis=1)
    df_plot.dropna(inplace=True)
    
    df_HC = df_plot[df_plot['Group'] == 0]
    df_pat = df_plot[df_plot['Group'] != 0]
    
    df_pat[df_pat.columns.difference(["Group"])].columns

    c, p = stats.spearmanr(df_pat['Dist'], df_pat[y_axis])
    r = r'$\rho$ = '
    if method == "Pearson":
        c, p = stats.pearsonr(df_pat['Dist'], df_pat[y_axis])
        r = 'r ='
    
    r_value = np.round(c, 2)
    p_value = np.round(p, 3)

    with _lock:
        fig, ax = plt.subplots(1,1,figsize=(4,4))
        sns.set(style='white', font_scale=1.5)
        g = sns.JointGrid(x=df_pat['Dist'], y=df_pat[y_axis])
        g = g.plot_joint(sns.regplot, color="#c54630")
        plt.xlabel("Anomaly score",size=28)
        plt.ylabel(y_axis,size=28)
        g = g.plot_marginals(sns.distplot, kde=False, bins=12, color="#c54630")
        g.fig.text(0.65, 0.22,r+ str(r_value)+'\np = '+str(p_value), fontsize=16) #add text
        #g.fig.text(0.44, 0.80,"Group", fontsize=22) #add text
        plt.tight_layout()
        g.savefig('figures/'+y_axis+'_'+savename+'.svg', dpi=200)
        st.write(g.fig)
        plt.close(fig)

