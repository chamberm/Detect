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

from utils import loader, explorer, launcher, reporter

#Top banner
script_dir = dirname(__file__) #<-- absolute dir the script is in
rel_path = "../ressources/banner2.png"
abs_file_path = join(script_dir, rel_path)
image = Image.open(abs_file_path)
st.image(image,use_column_width=True)

"""
# Detect 
A deep learning based anomaly detection framework for diffusion MRI Tractometry.

Author: Maxime Chamberland ([chamberm.github.io](https://chamberm.github.io)).
See the paper [here](https://www.nature.com/articles/s43588-021-00126-8). 

----
"""

def main():
    #LOAD data
    rel_path = "../ressources/demogA.csv"
    demog = join(script_dir, rel_path)
    df_demog = loader.load_csv(demog)
    
    rel_path = "../ressources/featuresA.xlsx"
    features = join(script_dir, rel_path)
    datasheet = loader.load_data(features)
    ################################

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
    tract_list = ['AF', 'ATR', 'CA', 'CC_1', 'CC_2', 'CC_3', 'CC_4', 'CC_5', 'CC_6', 'CC_7', 'CG',
                  'CST', 'FX', 'IFO', 'ILF', 'OR', 'SLF_I', 'SLF_II', 'SLF_III', 'UF', 'All']
    
    if st.checkbox('Show tract profiles'):
        plot_controls = st.checkbox('Plot Controls', True)
        plot_patients = st.checkbox('Plot Patients', True)
        show_indiv = st.checkbox('Show Individuals', False)
        tract_selection = st.selectbox("Choose a tract to visualize:", tract_list, 0)
        
        #plot profiles
        if (tract_selection.startswith('All')):      
            tract_selection = tract_list[:-1]
             
            for idx,t in enumerate(tract_selection):
                fig = explorer.plot_profile(df_data, df_demog, t, metric, plot_controls, plot_patients, group, show_indiv)
                st.write(fig)
        else:
            fig = explorer.plot_profile(df_data, df_demog, tract_selection, metric, plot_controls, plot_patients, group, show_indiv)
            st.write(fig)

    #Anomaly detection section
    #############################################
    st.header("2. Analysis section")
    options = ("Z-score", "PCA", "AutoEncoder")
    method = st.sidebar.radio("Method", options, 2)
    
    hemi = "Both"
    
    if method == "Z-score":
        use_all = st.checkbox("Use all", True)
        hemi_list = ("left", "right")
        if not use_all:
            hemi = st.radio("Hemisphere (if applicable))", hemi_list, 0)
    
    tract_init = []
    if method == "Z-score":
        if not use_all:
            uni_selection = st.selectbox("Choose a tract (univariate approach):", tract_list[:-1], 0)
            tract_profile = [uni_selection]
        else:
            tract_profile = tract_list[:-1]
    else:
        if (st.checkbox('Clear all')):
            tract_init = []
        else:
            tract_init = tract_list[:-1]
        tract_profile = st.multiselect("Choose tracts:", tract_list[:-1], tract_init)

    rep_value = 10
    if method == "SVM":
        rep_value = 10
    reps = st.sidebar.number_input('Iterations', min_value=1, max_value=100, value=rep_value)
    regress = False
    if st.sidebar.checkbox('Regress confound?',True):
        regress = not regress
        
    #input_threshold = st.sidebar.number_input("Anomaly threshold", 0.0, 10.0, input_threshold)
    st.write("Using ", method, "and the following tracts: ", ", ".join(tract_profile))
    
    #Analysis section
    #############################################
    result = "No results to display."
    if st.sidebar.button("Run"):
        if method == "SVM":
            AUC = launcher.svmachine(method, df_data, df_demog, regress, tract_profile, group, hemi, metric, reps)
            np.savetxt("tests/auc"+metric+method+title+".csv", AUC, delimiter=",")
        else:
            AUC, result, fpr, tpr = launcher.run(method, df_data, df_demog, regress, tract_profile, group, hemi, metric, reps)
    
            #Report section
            #############################################
            st.header("3. Report section")
            reporter.final_report(AUC, result, fpr, tpr, method, metric, group, title)
    
    #if st.sidebar.button("Save report"):
        #reporter.save(result, method)
        
    

if __name__ == '__main__':
    main()