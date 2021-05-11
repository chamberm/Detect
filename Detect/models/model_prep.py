from __future__ import division, print_function, absolute_import
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.preprocessing import  StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from models import Zscore, PCA, autoencoder

from matplotlib.backends.backend_agg import RendererAgg
_lock = RendererAgg.lock

class Model:
    def __init__(self, X_train, X_test, modeltype):
        self.x_train = X_train
        self.x_test = X_test
        self.modeltype = modeltype

    def get_train(self):
        return self.x_train
    
    def get_test(self):
        return self.x_test
    
    def run(self):
        if(self.modeltype == "Z-score"):
            return Zscore.run(self)
        elif(self.modeltype == "PCA"):
            return PCA.run(self)
        else:
            return autoencoder.run(self) 
        
    def run_once(self):
        return autoencoder.run_once(self)

def plotDistribution(d_train, xlim, ylim, label, method):
    with _lock:    
        fig, ax = plt.subplots(figsize=(12, 8))
        plt.title(method,size=46)
        sns.distplot(d_train,
                         bins = 10, 
                         kde= True,
                         norm_hist=True,
                         color = 'xkcd:purply',
                         kde_kws={"color": "xkcd:purply", "lw": 4});
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(label,size=36)
        ax.set_ylabel("Density estimate",size=36)
        ax.tick_params(labelsize=24)
        fig.tight_layout()
        fig.savefig('figures/'+method+'_distribution.png', dpi=200)
        st.write(fig)
        plt.close(fig)   
    
def evaluate(d_train, d_test, y_HC, y_PAT, method):
    anomaly = pd.DataFrame()
    anomaly['Dist']= np.abs(d_test)
    anomaly.dropna(inplace= True)
    anomaly['Group'] = pd.concat([y_HC['Group'], y_PAT['Group']])
    anomaly['ID'] = pd.concat([y_HC['ID'], y_PAT['ID']])
    st.write(anomaly)
    
    if(method == "Z-score"):
        label = 'Z-score (Z)'
        xlim = [-2,2]
        ylimDist = [0,1]
        ylimError = [0,2]
    elif(method == "PCA"):
        label = 'Mahalanobis distance (M)'
        xlim = [0.0,10]
        ylimDist = [0.0,1]
        ylimError = [0,10]
    else:
        label = 'Mean absolute error (MAE)'
        xlim = [0.05,0.25]
        ylimDist = [0,30]
        ylimError = [0.10,0.25]
        
    plotDistribution(d_train, xlim, ylimDist, label, method)
    
    with _lock:
        fig, ax = plt.subplots(1,1,figsize=(12, 8))
        ax.set_ylim(ylimError)
        ax.set_xlim((-1,len(d_test)))
        ax.set_xlabel("Individuals",size=28)
        ax.set_ylabel(label,size=28)
        ax.tick_params(labelsize=24)
        #ax.set_yticks(np.arange(0, ylimDist[1], step=0.25))
        #ax.set_yticks(np.arange(0, ylimDist[1], step=0.25))
        ax.set_title(method, size=36)

        rangeHC = np.arange(0,len(anomaly.loc[anomaly['Group'] == 0]))
        rangeAnom = np.arange(len(rangeHC),len(rangeHC)+len(anomaly.loc[anomaly['Group'] != 0]['Dist']))

        st.write("Group size:", len(rangeAnom), "controls, ", len(anomaly.loc[anomaly['Group'] != 0]['Dist']) ,"patients")
        ax.bar(rangeHC,anomaly.loc[anomaly['Group'] == 0]['Dist'], color='forestgreen', alpha=1, edgecolor="white", width=1.0, label="Controls")
        ax.bar(rangeAnom,anomaly.loc[anomaly['Group'] != 0]['Dist'], color='xkcd:eggplant purple', alpha=1, edgecolor="white", width=1.0,label="Patients")
        ax.legend(fontsize=24, loc='upper right')

        fig.tight_layout()
        st.write(fig)
        plt.close(fig)
    
    return anomaly
           
#Select tracts from dataframe based on user selection          
def select_features(method, df_data, tracts, hemi):
    if method == "Z-score":
        if any("CC" in t for t in tracts):
            hemi = ""
        X = df_data.loc[:, df_data.columns.str.startswith('Group') | 
                df_data.columns.str.startswith('ID') |
                df_data.columns.str.contains('|'.join([t+'_'+hemi for t in tracts]))]  
    else:
        X = df_data.loc[:, df_data.columns.str.startswith('Group') | 
                df_data.columns.str.startswith('ID') |
                df_data.columns.str.contains('|'.join(tracts))]
    
    return X
    
#Remove age or other effect
#TODO: Do it per tract instead of all measures at once.
def regress_confound(X_train_split, X_test_split,confound):
    #Init empty dataframe and setup regression model
    X_train_split_ageCorr = pd.DataFrame().reindex_like(X_train_split)
    #y_pred = pd.DataFrame().reindex_like(X_train_split)

    ## Now have to apply it to the test data too.
    X_test_split_ageCorr = pd.DataFrame().reindex_like(X_test_split)

    #y_test_pred = pd.DataFrame().reindex_like(X_test_split)

    regr = linear_model.LinearRegression()

    #fit a line between Age and mean Feature.
    #Then predict the Age corrected values.
    #Then Subtract those from the original measurements to obtain an Age-independant feature set.
    
    #x = np.array([age.loc[X_train_split.index]]).T
    dummies = pd.get_dummies(confound.sex)
    regress = pd.DataFrame()
    regress['age'] = confound['age']
    regress['gender'] = dummies.M
    x = regress[['age', 'gender']].loc[X_train_split.index].values.reshape(-1, 2) 
    
    y_1 = np.mean(X_train_split, axis = 1)
    globalMean = np.mean(y_1)
    
    regr.fit(x, y_1)
    y_pred = regr.predict(x)

    for i in np.arange(X_train_split.shape[0]):
        X_train_split_ageCorr.iloc[i] = X_train_split.iloc[i] - y_pred[i] +globalMean

    #if only 1 subject
   #xx = age.loc[X_test_split.index].values.reshape(1, -1)
    xx = regress[['age', 'gender']].loc[X_test_split.index].values.reshape(-1, 2) 
    #if multiple
    if (len(regress['age'].loc[X_test_split.index]) > 1):
        xx =  regress[['age', 'gender']].loc[X_test_split.index].values.reshape(-1, 2) 
    
    y_test_pred = regr.predict(xx)

    for i in np.arange(X_test_split.shape[0]):
        X_test_split_ageCorr.iloc[i] = X_test_split.iloc[i] - y_test_pred[i] +globalMean
    
    return X_train_split_ageCorr, X_test_split_ageCorr

def normalize_features(X_train_split, X_test_split, method):
    #No use, just to be able to warp back later
    scaler = MinMaxScaler()
    if method == "Z-score":
        scaler = StandardScaler()

    X_train_split[X_train_split.columns] = scaler.fit_transform(X_train_split[X_train_split.columns])
    X_test_split[X_test_split.columns] = scaler.transform(X_test_split[X_test_split.columns])

    return scaler, X_train_split, X_test_split

