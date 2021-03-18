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

from utils import loader, stats

#Top banner
script_dir = dirname(__file__) #<-- absolute dir the script is in
rel_path = "../ressources/banner2.png"
abs_file_path = join(script_dir, rel_path)
image = Image.open(abs_file_path)
st.image(image,use_column_width=True)

"""
# Relate 
A deep learning based anomaly detection framework for diffusion MRI Tractometry.

Author: Maxime Chamberland ([chamberm.github.io](https://chamberm.github.io)).
See the MedRxiv preprint [here](https://www.medrxiv.org/content/10.1101/2021.02.23.21252011v1).

----
"""

def main():
    parser = argparse.ArgumentParser(description="Anomaly detection.",
                                     epilog="Written by Maxime Chamberland.")
    parser.add_argument("--i", metavar="data_path", dest="data_file",
                        help="Measures (.csv)", required=True, 
                        type=abspath)
    parser.add_argument("--demog", metavar="demog_path", dest="demog_file",
                        help="file containing demographics (.csv)", required=True, 
                        type=abspath)
    args = parser.parse_args()

    #Load datasheets
    #############################################
    df_demog = loader.load_csv(args.demog_file)
    df_data = loader.load_csv(args.data_file)
    filename = args.data_file.rsplit('\\', 1)[-1]

    st.sidebar.subheader("File Uploader")
    up_demog = st.sidebar.file_uploader("Upload demographics", type="csv")
    if up_demog:
        df_demog = pd.read_csv(up_demog)
        
    up_data = st.sidebar.file_uploader("Upload measures", type="csv")
    if up_data:
        datasheet = loader.load_data(up_data)
    
    title = st.sidebar.text_input('Savename label', 'MY_ANALYSIS')
    
    options = ("Spearman", "Pearson")
    method = st.sidebar.radio("Method", options, 0)
    
    y_choices = df_demog[df_demog.columns.difference(["Group", "ID"])].columns
    y_axis = st.selectbox("Choose a clinical correlate below", y_choices, 0)
    
    stats.correlate(method, df_demog, df_data, y_axis, title, filename)
        
if __name__ == '__main__':
    main()