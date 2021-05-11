from __future__ import division, print_function, absolute_import
from sklearn.metrics import precision_recall_curve, roc_curve, auc, f1_score

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import seaborn as sns
import base64
from numpy import interp

from models import Zscore, PCA, autoencoder

from matplotlib.backends.backend_agg import RendererAgg
_lock = RendererAgg.lock

def get_csv_link(df, name):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    df.to_csv(name)
    csv = df.to_csv()
    b64 = base64.b64encode(
        csv.encode()
    ).decode()  # some strings <-> bytes conversions necessary here
    return f'<a href="data:file/csv;base64,{b64}" download="{name}"><input type="button" value="Download anomaly scores"></a>'

def get_txt_link(df, name):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    
    np.savetxt(name, df, delimiter=",")
    df = pd.DataFrame(df)
    csv = df.to_csv(index=False, header=False)
    
    b64 = base64.b64encode(
        csv.encode()
    ).decode()  # some strings <-> bytes conversions necessary here
    return f'<a href="data:file/csv;base64,{b64}" download="{name}"><input type="button" value="Download AUC"></a>'
   
def report_steps(curve, result, method, group, metric, plot=False):
    #st.write(result)
    result.dropna(inplace= True)
    y_true = result['Group']
    y_scores = result['Dist']
    no_skill = 0.50
    auc_score = 1.0
    
    with _lock:
        fig, ax = plt.subplots(1,1,figsize=(8, 8))

        #ROC
        ###############################################
        if curve == "ROC":
            fpr, tpr, thresholds = roc_curve(y_true, y_scores,group)
            auc_score = auc(fpr, tpr)

            ax.plot(fpr, tpr, color='navy',linewidth=3)
            ax.plot([0, 1], [0, 1], linestyle='--')

            x_label="False positive rate"
            y_label="True positive rate"
        #Precision recall
        ###############################################
        else:
            precision, recall, thresholds = precision_recall_curve(y_true, y_scores,group)
            f1score = 2 * (precision * recall) / (precision + recall)
            auc_score = auc(recall, precision)

            ax.plot(recall, precision, color='navy',linewidth=3)
            # calculate the no skill line as the proportion of the positive class
            no_skill = len(y_true[y_true==group]) / len(y_true)
            # plot the no skill precision-recall curve
            ax.plot([0, 1], [no_skill, no_skill], linestyle='--', label="random")

            x_label="Recall"
            y_label="Precision"
            ax.set_ylim(0,1)
            ax.set_ylim(0,1)

            fpr, tpr = precision, recall

        ax.set_xlabel(x_label,size=28)
        ax.set_ylabel(y_label,size=28)
        ax.tick_params(labelsize=24)

        ax.spines['left'].set_linewidth(3)
        ax.spines['bottom'].set_linewidth(3)

        ax.set_title('ROC', size=36)
        sns.despine()

        fig.tight_layout()
        if(plot):
            fig.savefig('figures/'+curve+'_'+metric+'_'+method+'.png', dpi=200)
        st.write(fig)
        plt.close(fig)

    st.write("AUC: ", np.round(auc_score,2))
    st.write("AUC random: ", np.round(no_skill,2))
    
    return auc_score, fpr, tpr

def writeCSV(ww, auc, metric, group, method, title):
    name = 'tests/scores'+'_'+metric+'_'+method+'_'+title+'.csv'
    st.markdown(get_csv_link(ww,name), unsafe_allow_html=True)
    
    name = "tests/auc"+'_'+metric+'_'+method+'_'+title+".csv"
    st.markdown(get_txt_link(auc,name), unsafe_allow_html=True)

def average_ROC(AUC, WW, fpr, tpr, method, metric, group, title):
    with _lock:
        fig, ax = plt.subplots(1,1,figsize=(8, 8))
        mean_fpr = np.linspace(0, 1, 100)
        tprs = []
        aucs = []

        c = "xkcd:purply"
        for i in range(len(AUC)):
            #plt.plot(fpr[i], tpr[i], lw=1, color = c, alpha=0.6)

            interp_tpr = interp(mean_fpr, fpr[i], tpr[i])
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(auc(fpr[i],tpr[i]))

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax.plot(mean_fpr, mean_tpr, color=c,
                label=r' Mean ROC (AUC = %0.3f $\pm$ %0.3f)' % (mean_auc, std_auc),
                lw=2, alpha=1)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color="xkcd:purply", alpha=.3,
                            label=r'$\pm$ 1 std. dev.')
        ax.plot([0, 1], [0, 1], linestyle='--')

        ax.tick_params(axis='y',labelsize=24)
        ax.tick_params(axis='x',labelsize=24)
        ax.spines['left'].set_linewidth(3)
        ax.spines['bottom'].set_linewidth(3)
        ax.set_xlabel("False positive rate",size=28)
        ax.set_ylabel('True Positive rate',size=28)
        ax.set_title(method + ' Mean ROC', size=36)
        sns.despine()
        fig.tight_layout()
        fig.savefig('figures/average_ROC_'+metric+'_'+method+'.png', dpi=200)
        st.write(fig)
        plt.close(fig)
    
def final_report(AUC, WW, fpr, tpr, method, metric, group, title):
    st.success("Average ROC AUC: " + str(np.round(np.mean(AUC), 2)) + " +/- " + str(np.round(np.std(AUC),2)))
    sns.set_style("white")
    #plot average ROC
    average_ROC(AUC, WW, fpr, tpr, method, metric, group, title)
    st.write(WW)
    st.warning("If you see NaNs in the above table, please increase the number of iterations to loop over all subjects.")
    
    with _lock:
        fig, ax = plt.subplots(1,1,figsize=(8, 8))
        my_pal = {0: "#5b437a", group: "#f1815f"}
        my_pal2 = {0: "#2d284b", group: "#f1815f"}
        dataPlot = WW.loc[(WW['Group'] == 0) | (WW['Group'] == group)]
        st.write(dataPlot)
        ax = sns.boxplot(x="Group", y="Dist", data=dataPlot, showfliers = False, palette=my_pal, linewidth=2)
        ax = sns.swarmplot(x="Group", y="Dist", data=dataPlot, color=".25", palette=my_pal2, alpha=0.8, linewidth=1)
        ax.set_xlabel("Groups",size=28)
        ax.set_ylabel('Scaled distance',size=28)
        ax.set_title(method, size=36)

        ax.tick_params(axis='y',labelsize=24)
        ax.tick_params(axis='x',labelsize=24)
        ax.spines['left'].set_linewidth(3)
        ax.spines['bottom'].set_linewidth(3)
        sns.despine()
        fig.tight_layout()
        fig.savefig('figures/AUE_mean_'+metric+'_'+method+'.png', dpi=200)
        st.write(fig)
        plt.close(fig)

    #preicison recall
    report_steps("Precision", WW, method, group, metric, True)
    writeCSV(WW, AUC, metric, group, method, title)
    
def save(result, method):
    if method == "Z-score":
        Zscore.save(result)
    elif method == "PCA":
        PCA.save(result)
    else:
        autoencoder.save(result)
        
def write_pval(x, x_hat, mae, p_along, p_overall, p_div, subject, metric, group, title, cols, once):
    data = [[subject, group.iloc[0], np.round(np.mean(mae), 3), p_overall]]

    dfpval = pd.DataFrame(data, index=[0], columns=['ID', 'Group', 'Error', 'p-val'])
    dfpval.to_csv('tests/p-val'+'_'+metric+'_'+title+'.csv', mode='a', header=once, index=False)

    p_along = np.insert(p_along, 0, 0, axis=0)
    p_along = np.insert(p_along, 0, 0, axis=0)
    dfvector = pd.DataFrame([p_along], index=[0], columns=cols)
    dfvector['ID'] = subject
    dfvector['Group'] = group.iloc[0]

    dfvector.to_csv('tests/anomaly-vector'+'_'+metric+'_'+title+'.csv', mode='a', header=once, index=False)
    
    return dfpval, dfvector
    
def filterSpurious(p_along):
    p_along_binary = np.zeros(len(p_along))
    for i in range(len(p_along)):
        if i == 0:
            if (p_along[i] > 0 and p_along[i+1] > 0):
                p_along_binary[i] = 1
        elif i == len(p_along)-1:
            if (p_along[i] > 0 and p_along[i-1] > 0):
                p_along_binary[i] = 1
        elif ((p_along[i] > 0 and p_along[i-1] > 0) or (p_along[i] > 0 and p_along[i+1] > 0)):
                p_along_binary[i] = 1
            
    return p_along_binary
    
def plot_features(x, x_hat, mae, p_along, p_overall, p_div, subject, metric, group, title, cols, once):
    st.success("Mean Absolute Error (MAE, unscaled): " + str(np.round(np.mean(mae), 3)))

    if (p_overall < max(0.01, (1/p_div))):
        st.success("p < "+str(np.round(1/p_div,3)))
    else:
        st.error("p = "+str(p_overall))
    sns.set_style("white")
    
    with _lock:
        fig, ax = plt.subplots(1,1,figsize=(24, 8))
        ax.legend(fontsize=14, loc='upper right')

        ax.plot(x_hat[0],color='#6a1596',label='Reconstructed',linewidth=4, linestyle="dashed", alpha=0.8)
        ax.plot(x[0],color='xkcd:burnt orange',label='Original',linewidth=4)

        p_along_binary = filterSpurious(p_along)

        ax.step(np.arange(0,len(p_along_binary)), p_along_binary*1.8*np.mean(x_hat), color="#b43486", linewidth=2, linestyle="dotted", alpha=0.5)
        ax.fill_between(np.arange(0,len(p_along_binary)),np.zeros(len(p_along_binary)),p_along_binary*1.8*np.mean(x_hat), alpha=0.1, 
                        edgecolor='#b43486', facecolor='#b43486', step="pre", label="Anomaly")

        ax.set_xlim((0,x_hat.shape[1]))
        ax.set_ylim((0,2*np.mean(x_hat)))
        ax.set_xlabel('Features',size=42)
        ax.set_ylabel(metric,size=42)
        ax.set_title(subject, size=48)
        ax.tick_params(labelsize=28)
        ax.set_xticks(range(0, len(x[0]), 20))
        ax.set_xticklabels(np.arange(0, len(x[0]), 20))
        ax.legend(fontsize=36, loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=True, shadow=True, ncol=3)

        ax.spines['left'].set_linewidth(3)
        ax.spines['bottom'].set_linewidth(3)
        sns.despine()

        fig.tight_layout()
        fig.savefig('figures/'+subject+'_profile_'+metric+'.png', dpi=200)
        st.pyplot(plt)

        dfpval, dfvector = write_pval(x, x_hat, mae, p_along, p_overall, p_div, subject, metric, group, title, cols, once)

        plt.close(fig)
    
    return dfpval, dfvector
    
        
