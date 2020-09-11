# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 00:49:59 2020

@author: Jie.Hu
"""


import pandas as pd
import numpy as np


# read data
df = pd.read_csv('seg_0910.csv', skipinitialspace=True)


# mtx
X = df.iloc[:, 1:26].values
y = df.iloc[:,26].values


''' MLR '''
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, classification_report
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression


clf_lr = LogisticRegression(multi_class = 'multinomial', random_state = 1337)
clf_lr.fit(X, y)

# Predicting the train set results
y_pred = clf_lr.predict(X)
accuracy_score(y, y_pred)


rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1337)
acc = cross_val_score(estimator = clf_lr, X = X, y = y, cv = rskf, scoring='accuracy')
acc.mean(), acc.std()


# KF n GS
parameters = {'solver': ['liblinear', 'lbfgs', 'sag', 'saga'], 
               'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

grid_search = GridSearchCV(estimator = clf_lr,
                           param_grid = parameters,
                           scoring='accuracy',
                           cv = 5,
                           n_jobs = -1)

grid_search = grid_search.fit(X, y)
grid_search.best_params_, grid_search.best_score_

# last step
clf_lr = LogisticRegression(multi_class = 'multinomial', 
                            solver = 'sag',
                            C = 0.1,
                            random_state = 1337)
clf_lr.fit(X, y)
y_pred = clf_lr.predict(X)
accuracy_score(y, y_pred)

print(classification_report(y, y_pred))


''' NB '''
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
clf_nb = MultinomialNB(alpha = 1)
# define evaluation procedure
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1337)
acc = cross_val_score(estimator = clf_nb, X = X, y = y, cv = rskf, scoring='accuracy')
acc.mean(), acc.std()


''' KNN '''
from sklearn.neighbors import KNeighborsClassifier
clf_knn = KNeighborsClassifier(n_neighbors = 50, p = 2, weights= 'distance')
# define evaluation procedure
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1337)
acc = cross_val_score(estimator = clf_knn, X = X, y = y, cv = rskf, scoring='accuracy')
acc.mean(), acc.std()



''' SVM '''
from sklearn.svm import SVC
clf_svc = SVC(random_state = 1337)
# define evaluation procedure
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1337)
acc = cross_val_score(estimator = clf_svc, X = X, y = y, cv = rskf, scoring='accuracy')
acc.mean(), acc.std()


# KF n GS
parameters = [{'kernel': ['linear'], 
               'C': [0.1, 1, 10], 
               'gamma': [0.1, 1, 10, 'auto', 'scale']}]
    
parameters = [{'kernel': ['poly'], 
               'C': [0.1, 1, 10], 
               'gamma': [0.1, 1, 10, 'auto', 'scale'], 
               'degree': [1,2,3]}]
               
parameters = [{'kernel': ['rbf'], 
               'C': [0.1, 1, 10], 
               'gamma': [0.1, 1, 10, 'auto', 'scale']}]


grid_search = GridSearchCV(estimator = clf_svc,
                           param_grid = parameters,
                           scoring='accuracy',
                           cv = 5,
                           n_jobs = -1)

grid_search = grid_search.fit(X, y)
grid_search.best_params_, grid_search.best_score_


# last step
clf_svc = SVC(kernel = 'poly',
              C = 0.1,
              gamma = 10, 
              degree = 1,
              random_state = 1337)
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1337)
acc = cross_val_score(estimator = clf_svc, X = X, y = y, cv = rskf, scoring='accuracy')
acc.mean(), acc.std()


''' LDA '''
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
clf_lda = LinearDiscriminantAnalysis(solver = 'lsqr', shrinkage = 0.15)
# define evaluation procedure
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1337)
acc = cross_val_score(estimator = clf_lda, X = X, y = y, cv = rskf, scoring='accuracy')
acc.mean(), acc.std()



''' QDA '''
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
clf_qda = QuadraticDiscriminantAnalysis(reg_param = 0.45)
# define evaluation procedure
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1337)
acc = cross_val_score(estimator = clf_qda, X = X, y = y, cv = rskf, scoring='accuracy')
acc.mean(), acc.std()
