![Detect](https://github.com/chamberm/Detect/blob/master/ressources/banner2.png)
# Detect
A browser-based anomaly detection framework for diffusion MRI using Tractometry. This repository contains the scripts used in [Chamberland et al. 2021](https://www.medrxiv.org/content/10.1101/2021.02.23.21252011v1). If using, please cite the following:
```
Chamberland, Maxime, Sila Genc, Chantal MW Tax, Dmitri Shastin, Kristin Koller, Erika P. Raven, Greg D. Parker, Khalid Hamandi, William P. Gray, and Derek K. Jones. 
"Detecting microstructural deviations in individuals with deep diffusion MRI tractometry." medRxiv (2021).
```
# Live demo (Browser)
* :star2: Click here for [Detect](https://share.streamlit.io/chamberm/detect/Detect/detect-demo.py) (simply hit Run!)
* :star2: Click here for [Inspect](https://share.streamlit.io/chamberm/detect/Detect/inspect-demo.py) (simply hit Run!)  
Note: Streamlit servers may sometimes be overloaded - in this case, try refreshing the page.

# Install (to run locally)
This requires: 
* [Streamlit](https://www.streamlit.io/)
* [Tensorflow](https://www.tensorflow.org/)
* [Keras](https://keras.io/)
* [Scikit-learn](https://scikit-learn.org/stable/)

as well as plotting libraries described in the requirements.txt file. Run:
```
git clone https://github.com/chamberm/Detect
cd Detect

#Create a new environment
conda create --name Detect python=3.7
pip install -r requirements.txt
pip install -e .
```
# Data format
The input demographic data that consists of comma-separated values (.csv) where each row represents a subject (ID). Example demographics columns include: group, age, gender or clinical scores. 

The microstructural tractometry data format consist of a .xlsx spreadsheet, where each sheet represents a dMRI measure (e.g., FA, MD, AFD, RISH, etc.). As per the demographic data, subjects are stacked individually on each rows. The first column denotes the ID of each subject. The remaining columns follow the following convention: *BUNDLE_HEMI_SECTION* where *BUNDLE* is the white matter bundle of interest, *HEMI* is the hemisphere (i.e., left or right and void for commissural tracts), and *SECTION* is the along-tract portion (e.g., from 1 to 20). 

The framework offers three main script: **Detect**, **Inspect** and **Relate**. Both Detect and Inspect scripts allow the visualization of tract profiles. Finally, Relate is a simple visual interface to correlate the anomaly scores obtained by the previous commands with clinical scores.

![AE](https://github.com/chamberm/Detect/blob/master/ressources/AE.png)

# Example Usage
## Detect:
Detect allows for group comparisons using cross-validated AUCs computed over N iterations, by means of Z-score, PCA or AE. The output is a robust, bootstrapped anomaly score for each subject.
```
python ./bin/detect.py --i DATA.xlsx --demog DEMOG.csv
```
0. Load your dataset
1. Input a savename on the left (e.g., my_analysis).
2. Choose a metric from the dropdown menu (e.g., RISH0).
3. Choose a patient group (when multiple groups are present).
4. Choose an anomaly detection method (e.g., AutoEncoder).
5. Choose the number of iterations to bootstrap over (e.g., 50, 100).
6. You can visualize the tract-profiles using the options on the main page (remember to turn them off before lauching the analysis).
7. Choose the desired features (tracts) to include in the analysis.
8. Hit the Run button on the bottom left panel.

The global anomaly scores of each subject will be saved as: *tests/scores_METRIC_METHOD_SAVENAME.csv* 
and the ROC AUC under *tests/auc_METRIC_METHOD_SAVENAME.csv*


## Inspect:
On the other hand, Inspect allows the user to select a single subject and to compare it with the rest of the population. Here, anomalies in the features are highlighted using a leave-one-out cross-validation approach.
```
python ./bin/inspect.py --i DATA.xlsx --demog DEMOG.csv
```
0. Load your dataset
1. Input a savename on the left (e.g., my_analysis).
2. Choose a metric from the dropdown menu (e.g., RISH0).
3. Choose a patient group (when multiple groups are present).
4. Choose subject to inspect.
5. You can visualize the tract-profiles using the options on the main page. Remember to turn them off before lauching the analysis.
6. If you desire, the Run all option will iterate over all subjects to test them against the healthy controls.
8. Hit the Run button on the bottom left panel.

The overall anomaly score of that subject will be saved as: *tests/p-val_METRIC_SAVENAME.csv* 
and the along-tract profile figure under *figures/SUBJECT_profile_METRIC.png*.

**Note**: the **examples** directory contains notebooks with code used to generate the results and figures from the paper.

# Disclaimer
Please note that Detect is for research use only. 

# Author
Maxime Chamberland [Website](https://chamberm.github.io/)

