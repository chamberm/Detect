from __future__ import division, print_function, absolute_import

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from sklearn.decomposition import PCA
import seaborn as sns

def save(model):
    pass

def covar_matrix(data, verbose=False):
    covariance_matrix = np.cov(data, rowvar=False)
    if is_pos_def(covariance_matrix):
        inv_covariance_matrix = np.linalg.inv(covariance_matrix)
        if is_pos_def(inv_covariance_matrix):
            return covariance_matrix, inv_covariance_matrix
        else:
            print("Error: Inverse of Covariance Matrix is not positive definite!")
    else:
        print("Error: Covariance Matrix is not positive definite!")
        
def MahalanobisDist(inv_cov_matrix, mean_distr, data, verbose=False):
    inv_covariance_matrix = inv_cov_matrix
    vars_mean = mean_distr
    diff = data - vars_mean
    md = np.zeros(len(diff))
    for i in range(len(diff)):
        md[i] = np.sqrt(diff[i].dot(inv_covariance_matrix).dot(diff[i]))
    return md

def MD_detectOutliers(dist, extreme=False, verbose=False):
    k = 3. if extreme else 2.
    threshold = np.mean(dist) * k
    outliers = []
    for i in range(len(dist)):
        if dist[i] >= threshold:
            outliers.append(i)  # index of the outlier
    return np.array(outliers)

def MD_threshold(dist, extreme=False, verbose=False):
    k = 3. if extreme else 2.
    threshold = np.mean(dist) * k
    return threshold

def is_pos_def(A):
    if np.allclose(A, A.T):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False
    
def run(model):
    X_train = model.get_train()
    X_test = model.get_test()
    pca = PCA(n_components=3, svd_solver= 'full')
    X_train_PCA = pca.fit_transform(X_train)
    X_test_PCA = pca.transform(X_test)
    
    X_train_PCA = pd.DataFrame(X_train_PCA)
    X_train_PCA.index = X_train.index
    data_train = X_train_PCA.values
    
    X_test_PCA = pd.DataFrame(X_test_PCA)
    X_test_PCA.index = X_test.index
    data_test = X_test_PCA.values
    
    cov_matrix, inv_cov_matrix  = covar_matrix(data_train)
    mean_distr = np.mean(data_train, axis=0)
    
    Mob_test = MahalanobisDist(inv_cov_matrix, mean_distr, data_test, verbose=False)
    M_test = np.squeeze(pd.DataFrame(data=Mob_test, index=X_test.index))
    
    Mob_train = MahalanobisDist(inv_cov_matrix, mean_distr, data_train, verbose=False)
    M_train = np.squeeze(pd.DataFrame(data=Mob_train, index=X_train.index))

    return M_train, M_test
    