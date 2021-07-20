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

#Display info on demographics
def display_demog(df_demog):
    st.write(df_demog)
    st.write(df_demog.shape)
    st.write("Subjects with missing values:", df_demog[df_demog.isnull().T.any().T])

#Display info on dataset
def display_data(df_data):
    st.write(df_data)
    st.write(df_data.shape)
    st.write("Subjects with missing values:", df_data[df_data.isnull().T.any().T])
    
def plot_profile(df_data, df_demog, tract_profile, metric, plot_controls, plot_patients, group, show_indiv):
    y = df_demog['Group']
    HC = df_data[df_data['Group'] == 0]
    PATIENTS = df_data[df_data['Group'] == group].reset_index(drop=True)
    HC.index = range(len(HC))
    PATIENTS.index = range(len(PATIENTS))
    sigma = 1.5

    with _lock:
        #when plotting a single bilateral tract 
        if not (tract_profile.startswith('CC') or tract_profile.startswith('CA')):
            sides = ['left', 'right']
            fig, ax = plt.subplots(2,1,figsize=(12, 16), squeeze=False)
        else:
            sides = [""]
            fig, ax = plt.subplots(1,1,figsize=(12, 8), squeeze=False)

        for idx,hemi in enumerate(sides):
            filter_HC = [col for col in HC if col.startswith(tract_profile+'_'+hemi)]
            filter_PAT = [col for col in PATIENTS if col.startswith(tract_profile+'_'+hemi)]
            noOfFeatures = len(filter_HC)
            mean_PAT = np.mean(PATIENTS[filter_PAT])
            mean_HC = np.mean(HC[filter_HC])
            std_PAT = stats.sem(PATIENTS[filter_PAT])
            std_HC = np.std(HC[filter_HC])*1.65

            #Individual profiles
            if show_indiv:
                for i in range(len(HC)):
                    if plot_controls:
                        indiv = gaussian_filter1d(HC[filter_HC].T[i], sigma=sigma)
                        ax[idx,0].plot(indiv, color='xkcd:purply',alpha=0.2)

                if plot_patients:
                    for i in range(len(PATIENTS)):
                        if plot_patients: 
                            rgb = np.random.rand(3,)
                            indiv = gaussian_filter1d(PATIENTS[filter_PAT].T[i], sigma=sigma)
                            ax[idx,0].plot(indiv, c=rgb, alpha=1, lw=4, label=df_demog.loc[i].ID)

            #Patients only group mean profiles          
            if plot_patients:
                pass
                #ax[idx,0].plot(gaussian_filter1d(mean_PAT,sigma=sigma),color='crimson',label='Patients',linewidth=4)
                #ax[idx,0].fill_between(range(len(mean_PAT)),gaussian_filter1d(mean_PAT-std_PAT, sigma=sigma), 
                                     #gaussian_filter1d(mean_PAT+std_PAT, sigma=sigma), 
                                     #alpha=0.5, edgecolor='crimson', facecolor='red')

            #Controls only group mean profiles
            if plot_controls:
                ax[idx,0].plot(gaussian_filter1d(mean_HC,sigma=sigma),color='xkcd:purply',label='Controls',linewidth=4)
                ax[idx,0].fill_between(range(len(mean_HC)),gaussian_filter1d(mean_HC-std_HC, sigma=sigma), 
                                     gaussian_filter1d(mean_HC+std_HC, sigma=sigma), 
                                     alpha=0.5, edgecolor='xkcd:purply', facecolor='#5b437a')

            #format
            ax[idx,0].set_xticks(range(0, noOfFeatures, 2))
            ax[idx,0].set_xticklabels(np.arange(0, noOfFeatures, 2))
            ax[idx,0].tick_params(labelsize=36)
            ax[idx,0].set_xlim(0, noOfFeatures-1)
            ax[idx,0].set_ylabel(metric, size=42)
            ax[idx,0].set_xlabel('Position along tract', size=42)
            ax[idx,0].set_title(hemi.upper()+' '+tract_profile,size=36)
            ax[idx,0].legend(loc='upper center',fontsize=24, bbox_to_anchor=(0.5, -0.30),
              fancybox=True, shadow=True, ncol=3)  

        fig.tight_layout()
        fig.savefig('figures/'+metric+'_'+tract_profile+'_profile.svg', dpi=200)
        plt.close(fig)

    return fig