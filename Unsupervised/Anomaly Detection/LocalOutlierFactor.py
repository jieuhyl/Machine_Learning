# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 11:58:57 2020

@author: Jie.Hu
"""



import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from pyod.utils.data import generate_data


contamination = 0.1  # percentage of outliers
n_train = 200  # number of training points
n_test = 100  # number of testing points

# Generate sample data
X_train, y_train, X_test, y_test = \
    generate_data(n_train=n_train,
                  n_test=n_test,
                  n_features=2,
                  contamination=contamination,
                  random_state=42)

# train LocalOutlierFactor
clf_name = 'LOF'
clf = LocalOutlierFactor(n_neighbors=3, novelty=False)
clf.fit(X_train)

# get the prediction labels and outlier scores of the training data
y_train_pred = clf.predict(X_train)  # binary labels (0: inliers, 1: outliers)
y_train_scores = clf.decision_function(X_train)   # raw outlier scores


# get the prediction on the test data, cannot predict train data(normal)
clf = LocalOutlierFactor(n_neighbors=3, novelty=True)
clf.fit(X_train)
y_test_pred = clf.predict(X_test)  # outlier labels (0 or 1)
y_test_scores = clf.decision_function(X_test)   # outlier scores
 

# Step 2: Determine the cut point
import matplotlib.pyplot as plt
plt.hist(y_test_scores, bins='auto')  
plt.title("Histogram with LOF Anomaly Scores")
plt.show()


test_scores  = pd.DataFrame({'Scores': y_test_scores,
                             'Labels': y_test_pred})
pd.DataFrame({'Outliers': test_scores.groupby('Labels').get_group(-1).Scores,
              'Inlierss': test_scores.groupby('Labels').get_group(1).Scores}).plot.hist(stacked=True)
