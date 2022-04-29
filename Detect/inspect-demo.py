#!/usr/bin/env python

from __future__ import division, print_function, absolute_import

import importlib
import argparse
import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from os.path import join, dirname, abspath
import seaborn as sns

from utils import loader, explorer, inspector, reporter

#Top banner
script_dir = dirname(__file__) #<-- absolute dir the script is in
rel_path = "../ressources/banner2.png"
abs_file_path = join(script_dir, rel_path)
image = Image.open(abs_file_path)
st.image(image,use_column_width=True)

"""
# Inspect 
A deep learning based anomaly detection framework for diffusion MRI Tractometry.

Author: Maxime Chamberland ([chamberm.github.io](https://chamberm.github.io)).
See the paper [here](https://www.nature.com/articles/s43588-021-00126-8).

----
"""

def main():
    #Load datasheets
    #############################################
    rel_path = "../ressources/demog-short.csv"
    demog = join(script_dir, rel_path)
    df_demog = loader.load_csv(demog)
    
    rel_path = "../ressources/features-short.xlsx"
    features = join(script_dir, rel_path)
    datasheet = loader.load_data(features)

    st.sidebar.subheader("File Uploader")
    up_demog = st.sidebar.file_uploader("Upload demographics", type="csv")
    if up_demog:
        df_demog = pd.read_csv(up_demog)
        
    up_data = st.sidebar.file_uploader("Upload profiles", type="xlsx")
    if up_data:
        datasheet = loader.load_data(up_data)
    
    title = st.sidebar.text_input('Savename', 'MY_ANALYSIS')
    
    #Default metrics and groups to work with
    metric = st.sidebar.selectbox("Choose a metric below", list(datasheet.keys()), 0)
    df_data = loader.combine_demog_and_data(df_demog, datasheet, metric)
    group_ids = df_demog.Group[df_demog.Group != 0].unique()
    group = st.sidebar.selectbox("Choose a patient group below", group_ids, 0)
    subjs_ids = df_demog.ID
    subject = st.sidebar.selectbox("Choose a subject below", subjs_ids, 1)
    
    #Display datasheets
    #############################################
    st.header("1. Visualisation section")
    sns.set(font_scale=1.3)
    if st.checkbox('Show demographics',False):
        st.write("Demographics datasheet")
        explorer.display_demog(df_demog)
        
    if st.checkbox('Show dataset',False):
        st.write("Tract-profiles datasheet")
        explorer.display_data(df_data)
    
    #Data exploration section
    #############################################
    #TODO automatically detect tracts
    tract_list = ['AF_left', 'AF_right','ATR_left','ATR_right', 'CA', 'CC_1', 'CC_2', 'CC_3', 'CC_4', 'CC_5', 'CC_6', 'CC_7', 'CG_left', 'CG_right', 
                  'CST_left', 'CST_right', 'FX_left', 'FX_right', 'IFO_left', 'IFO_right', 'ILF_left', 'ILF_right', 'OR_left', 'OR_right', 'SLF_I_left',
                  'SLF_II_left', 'SLF_III_left', 'SLF_I_right', 'SLF_II_right', 'SLF_III_right', 'UF_left', 'UF_right']
    
    tract_list_uni = ['AF', 'ATR', 'CA', 'CC_1', 'CC_2', 'CC_3', 'CC_4', 'CC_5', 'CC_6', 'CC_7', 'CG',
                  'CST', 'FX', 'IFO', 'ILF', 'OR', 'SLF_I', 'SLF_II', 'SLF_III', 'UF', 'All']
    
    if st.checkbox('Show tract profiles'):
        plot_controls = st.checkbox('Plot Controls', True)
        plot_patients = st.checkbox('Plot Patients', True)
        show_indiv = st.checkbox('Show Individuals', False)
        tract_selection = st.selectbox("Choose a tract to visualize:", tract_list_uni, 0)
        
        #plot profiles
        if (tract_selection.startswith('All')):      
            tract_selection = tract_list_uni[:-1]
             
            for idx,t in enumerate(tract_selection):
                fig = explorer.plot_profile(df_data, df_demog, t, metric, plot_controls, plot_patients, group, show_indiv)
                st.write(fig)
        else:
            fig = explorer.plot_profile(df_data, df_demog, tract_selection, metric, plot_controls, plot_patients, group, show_indiv)
            st.write(fig)

    #Anomaly detection section
    #############################################
    st.header("2. Analysis section")
    hemi = "Both"
    tract_init = []
    
    tract_list_demo = ['AF_left','CC_4', 'CC_5', 'CST_right', 'IFO_left', 'ILF_left',
                       'OR_left','SLF_I_right','UF_left']
    
    if (st.checkbox('Clear all')):
        tract_init = []
        tract_profile = st.multiselect("Choose tracts:", tract_list, tract_init)
    else:
        tract_init = tract_list_demo
        tract_profile = st.multiselect("Choose tracts:", tract_list, tract_init)

    regress = False
    if st.sidebar.checkbox('Regress confound?',True):
        regress = not regress
        
    #input_threshold = st.sidebar.number_input("Anomaly threshold", 0.0, 10.0, input_threshold)
    st.write("Using the following tracts: ", ", ".join(tract_profile))
    
    LOOCV = False
    #if st.sidebar.checkbox('Run all subjects?',False):
        #LOOCV = not LOOCV
    
    pop = [subject]
    #if LOOCV:
        #pop = subjs_ids
        
    #Analysis section
    #############################################
    result = "No results to display."
    finalpval = pd.DataFrame()
    finalvector = pd.DataFrame()
    if st.sidebar.button("Run"):
        
        once = True
        for s in pop:
            x, x_hat, mae, p_along, p_overall, p_div = inspector.run(s, df_data, df_demog, regress, tract_profile, hemi, metric)
            cur_group = df_demog.loc[df_data['ID'] == s, 'Group']
            #Report section
            #############################################
            st.header("3. Report section")
            X = df_data.loc[:, df_data.columns.str.startswith('Group') | 
                df_data.columns.str.startswith('ID') |
                df_data.columns.str.contains('|'.join(tract_profile))]
            dfpval, dfvector = reporter.plot_features(x, x_hat, mae, p_along, p_overall, p_div, s, metric, cur_group, title, X.columns, once)
            
            if once:
                finalpval = dfpval
                finalvector = dfvector
                
            finalpval.append(dfpval)
            finalvector.append(dfvector)   
            once = False
        
        name = 'tests/p-val'+'_'+metric+'_'+title+'.csv'
        st.markdown(reporter.get_csv_link(finalpval,name), unsafe_allow_html=True)
        name = 'tests/reconstructed-features'+'_'+metric+'_'+title+'.csv'
        st.markdown(reporter.get_csv_link_to_xhat(finalvector,name), unsafe_allow_html=True)
            
    #if st.sidebar.button("Save report"):
        #reporter.save(x_hat)
        
if __name__ == '__main__':
    main()