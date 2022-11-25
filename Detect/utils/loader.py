from __future__ import division, print_function, absolute_import

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import openpyxl

#Load demographics
@st.cache
def load_csv(file):
    data = pd.read_csv(file)
    return data

#Load data
@st.cache
def load_data(file):
    xls = pd.ExcelFile(file)
    sheet_to_df_map = {}
    for sheet_name in xls.sheet_names:
        sheet_to_df_map[sheet_name] = xls.parse(sheet_name)
        sheet_to_df_map[sheet_name].columns = ['ID']+[str(col) for col in sheet_to_df_map[sheet_name].columns[1:]]
    return sheet_to_df_map

#Combine demographics with a selected metric
@st.cache
def combine_demog_and_data(df_demog, datasheet, metric):
    #Concat all features
    df_data = pd.concat([df_demog['Group'], datasheet[metric]], axis=1).drop_duplicates().reset_index(drop=True)

    #Drop SID multiplicated
    df_data = df_data.loc[:,~df_data.columns.duplicated()]
    return df_data