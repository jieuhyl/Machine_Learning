# -*- coding: utf-8 -*-
"""
Created on Fri May  1 01:39:57 2020

@author: Jie.Hu
"""


import numpy as np
from sklearn.datasets import make_classification
import time

''' 7: AdaBoost '''
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, classification_report
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, GridSearchCV, train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, weights=[0.5], random_state=1337)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, shuffle=True, random_state=1337)
#train_df, valid_df = train_test_split(train_df, test_size=0.2, random_state=1337, stratify=True, shuffle=True)


# define the model
lr = LogisticRegression()
nb = GaussianNB()
dt = DecisionTreeClassifier(random_state = 1337)

clf_ada = AdaBoostClassifier(base_estimator=dt)
# evaluate the model
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1337)
acc = cross_val_score(estimator = clf_ada, X = X_train, y = y_train, cv = cv, scoring='f1')
acc.mean(), acc.std()


parameters = {'algorithm':['SAMME', 'SAMME.R'],
              'n_estimators':[50,100,200],
              'learning_rate':[0.01,0.1,1]}
                                       
grid_search = GridSearchCV(estimator = clf_ada,
                           param_grid = parameters,
                           scoring='f1',
                           cv = cv,
                           n_jobs = -1)
start_time = time.time()
grid_search = grid_search.fit(X_test, y_test)
print('Training time: {} minutes'.format(round((time.time() - start_time)/60, 2)))
grid_search.best_params_, grid_search.best_score_


# last step
clf_ada = AdaBoostClassifier(dt,
                             algorithm = 'SAMME',
                             n_estimators = 100, 
                             learning_rate = 0.1,
                             random_state= 1337 )
clf_ada.fit(X_train, y_train)
y_pred = clf_ada.predict(X_test)

acc = cross_val_score(estimator = clf_ada, X = X_test, y = y_test, cv = cv, scoring='f1')
acc.mean(), acc.std()

print(classification_report(y_test, y_pred))