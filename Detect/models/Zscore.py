from __future__ import division, print_function, absolute_import

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import seaborn as sns

def save(model):
    pass

def run(model):
    X_train = model.get_train()
    X_test = model.get_test()

    meanProfile = np.mean(X_train)
    stdProfile = np.std(X_train)

    zscoreTest = np.mean((X_test - meanProfile)/stdProfile, axis =1)
    zscoreTrain = np.mean((X_train - meanProfile)/stdProfile, axis =1)
    return zscoreTrain, zscoreTest
    
    