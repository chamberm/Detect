from __future__ import division, print_function, absolute_import

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import seaborn as sns
import random
from sklearn.model_selection import RepeatedKFold, train_test_split, cross_val_score, StratifiedKFold, RepeatedStratifiedKFold, GridSearchCV
from sklearn import svm, preprocessing
from models import PCA, autoencoder, model_prep
from sklearn.preprocessing import  StandardScaler, MinMaxScaler
from models.model_prep import Model
from utils import reporter
from sklearn.pipeline import make_pipeline
from sklearn.metrics import precision_recall_curve, roc_curve, auc, f1_score

def svmachine(method, df_data, df_demog, regress, tracts, group, hemi, metric, reps):

    #1 Select features
    X = model_prep.select_features(method, df_data, tracts, hemi)
    
    dis = df_demog.apply(lambda x: True if x['Group'] == group else False , axis=1)
    hcs = df_demog.apply(lambda x: True if x['Group'] == 0 else False , axis=1)

    # Count number of True in series and find ratio of HC/PAT for splitting
    numOfdis = len(dis[dis == True].index)
    numOfhcs = len(hcs[hcs == True].index)
    ratio = numOfdis/(numOfhcs+numOfdis)
    st.write("Ratio subjects/controls:", np.round(ratio,2))
    
    X = X[(X['Group'] == 0) | (X['Group'] == group)]
    y = X[['Group']]
    X = X.drop(['Group', 'ID'], axis=1)
    
    #scaler = MinMaxScaler(feature_range=(0, 1))
    #X[X.columns] = scaler.fit_transform(X[X.columns])
    param_grid = [{'kernel': ['rbf'], 
                   'gamma': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 
                     'C': [1, 10, 100, 1000]}]
    grid_search = GridSearchCV(svm.SVC(class_weight={0:ratio, group:1-ratio}), param_grid, scoring = "roc_auc")
    
    scores = []
    #best_svc = svm.SVC(kernel='linear', C=1, class_weight={0:ratio, group:1-ratio})
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=reps, random_state=42)
    for train_index, test_index in cv.split(X,y):
        X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
        if(regress):
            if'sex' in df_demog and 'age' in df_demog:
                X_train, X_test = model_prep.regress_confound(X_train, 
                                                       X_test, df_demog)
            else:
                st.error("No age or sex information found. Skipping regression step.")
            
        scaler, X_train, X_test = model_prep.normalize_features(X_train, X_test,method)
        #best_svc.fit(X_train, y_train.values.ravel())
        grid_search.fit(X_train, y_train.values.ravel())
        y_pred = grid_search.predict(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, y_pred,group)
        auc_score = auc(fpr, tpr)
        #st.write(auc_score)
        scores.append(auc_score)
    
    st.success("Mean AUC: %0.2f (+/- %0.2f)" % (np.round(np.mean(scores), 2), np.round(np.std(scores),2)))
    return scores

def run(method, df_data, df_demog, regress, tracts, group, hemi, metric, reps):
    st.warning("Computing permutations ... estimated time: " + str(np.round(len(df_demog)*2/60,2)) + " minutes.")
    
    if 'sex' not in df_demog and 'age' not in df_demog:
        st.error("No age or sex information found. Skipping regression step.")
    
    #1 Select features
    X = model_prep.select_features(method, df_data, tracts, hemi)
    # Get a bool series representing which row satisfies the condition i.e. True for
    # row in which value of group == 0
    #st.write(X)
    dis = df_demog.apply(lambda x: True if x['Group'] == group else False , axis=1)
    hcs = df_demog.apply(lambda x: True if x['Group'] == 0 else False , axis=1)

    # Count number of True in series and find ratio of HC/PAT for splitting
    numOfdis = len(dis[dis == True].index)
    numOfhcs = len(hcs[hcs == True].index)
    ratio = numOfdis/numOfhcs
    
    #To accumulate error Distances
    DISTS = np.zeros(len(X))
    countInserts = np.zeros(len(X))
    count=0
    
    #Separate HC from PATIENTS
    HC = X[X['Group'] == 0]
    y_HC = HC[['Group', 'ID']]
    PATIENTS = X[X['Group'] == group]
    y_PAT = PATIENTS[['Group', 'ID']]
    
    ##2 HERE, basically split the data into train and Val (split_size*repeats times) into equal size of HC/PAT.
    #split_size = int(np.ceil((float(numOfhcs)/numOfdis)))
    split_size = 5
    #st.write (split_size, numOfhcs, numOfdis)
    repeats = reps
    #random_state = 12883823
    #random_state=42
    #rkf = RepeatedKFold(n_splits=split_size, n_repeats=repeats, random_state=random_state)
    
    AUC = np.zeros(repeats)
    tpr = []
    fpr = []
    #for train_idx, test_idx in rkf.split(HC,y_HC):
    for r in range(repeats):
        st.write("Iteration", r+1 , "of", repeats)
        X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(HC, y_HC, test_size=min(0.2,ratio))
        #X_train_split, X_test_split = HC.iloc[train_idx], HC.iloc[test_idx]
        #y_train_split, y_test_split = y_HC.iloc[train_idx], y_HC.iloc[test_idx]
        #Select subset of patients
        patients_subset_ids = np.array(random.sample(list(PATIENTS.index), min(len(X_test_split), len(PATIENTS))))

        #Cating the HC test with the PATIENTS
        X_test_split = pd.concat([X_test_split,PATIENTS.loc[patients_subset_ids]])
        y_test_split = pd.concat([y_test_split, y_PAT.loc[patients_subset_ids]])
        
        X_train_split = X_train_split.drop(['Group', 'ID'], axis=1)
        X_test_split = X_test_split.drop(['Group', 'ID'], axis=1)
        
        
        #4 Normalize features
        if method != "Z-score":
            scaler, X_train_split, X_test_split = model_prep.normalize_features(X_train_split, X_test_split, method)
        else:
            X_train_split, X_test_split = X_train_split, X_test_split
            
            
        #3 Linear regression of confound
        if(regress):
            if'sex' in df_demog and 'age' in df_demog:
                X_train, X_test = model_prep.regress_confound(X_train_split, 
                                                       X_test_split, df_demog)
    
        #5 Anomaly detection method
        if method == "Z-score":
            model = Model(X_train, X_test, "Z-score")
        elif method == "PCA":
            model = Model(X_train, X_test, "PCA")
        else:
            model = Model(X_train, X_test, "Autoencoder")

        #6 Run 
        d_train, d_test = model.run()
        DISTS[d_test.index] = DISTS[d_test.index] + d_test.values
        countInserts[d_test.index] += 1
        
        #7 Evaluate 
        result = model_prep.evaluate(d_train, d_test, y_HC, y_PAT, method)
        AUC[count], f, t = reporter.report_steps("ROC", result, method, group, metric, False)
        tpr.append(t)
        fpr.append(f)
        count = count + 1
    
    #Assemble results / Aggregate and plot mean/median distributions 
    np.seterr(divide='ignore', invalid='ignore')
    DISTS /= countInserts
    WW = pd.DataFrame()
    WW['ID'] = X['ID']
    WW['Group'] = X['Group']
    WW['Dist'] = DISTS
    
    return AUC, WW, fpr, tpr