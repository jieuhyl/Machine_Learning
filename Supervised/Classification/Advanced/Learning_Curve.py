# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 01:34:01 2020

@author: Jie.Hu
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
import warnings
warnings.filterwarnings('ignore') 
from scipy import stats
from scipy.stats import norm, skew 
import time
from sklearn.model_selection import learning_curve


# read data
df = pd.read_csv('da_0420.csv', skipinitialspace=True)

# check missing again
df.isnull().values.sum()
df.isnull().sum()/df.shape[0]
missing_ratio = df.isnull().sum() / len(df)
df.columns.tolist()


# 
df['QPOSTINT'].value_counts(normalize = True)
df['QPOSTINT'] = df['QPOSTINT'].apply(lambda x: 1 if x==1 else 0)

def mapping(x):
    if x==1:
        val = 1
    else:
        val = 0
    return val
   
df.iloc[:,2:] = df.iloc[:,2:].applymap(mapping)


#df[['QFANSHIPr1','QSHOW_ELEMENTS_r13', 'QSHOW_ELEMENTS_r14']].groupby(['QFANSHIPr1']).agg(['mean', 'count'])


''' model '''
# get mtx
X = df.iloc[:, 2:].values
y = df.iloc[:,1].values




''' 5: Decision Tree'''
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, classification_report
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
clf_dt = DecisionTreeClassifier(random_state = 1337)
clf_dt.fit(X, y)


cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1337)
acc = cross_val_score(estimator = clf_dt, X = X, y = y, cv = cv, scoring='f1')
acc.mean(), acc.std()


parameters = {'criterion':['gini', 'entropy'],
              'max_depth':[3,4,5],
              'max_features':['auto', 'sqrt', 'log2'],
              'min_samples_leaf':[1,2,3,4,5],
              'min_samples_split':[2,4,6,8,9,10],
              'class_weight':[{0:3,1:1}, {0:2,1:1}, {0:1,1:1}, {0:1,1:2}, {0:1,1:3}, 'balanced']}
                                       
grid_search = GridSearchCV(estimator = clf_dt,
                           param_grid = parameters,
                           scoring='f1',
                           cv = cv,
                           n_jobs = -1)
start_time = time.time()
grid_search = grid_search.fit(X, y)
print('Training time: {} minutes'.format(round((time.time() - start_time)/60, 2)))
grid_search.best_params_, grid_search.best_score_


# last step
clf_dt = DecisionTreeClassifier(criterion = 'gini',
                                max_depth = 5, 
                                max_leaf_nodes = 10,
                                max_features = 'auto',
                                min_samples_leaf = 2,
                                min_samples_split = 8,
                                class_weight = {0: 1, 1: 3},
                                random_state= 1337 )
clf_dt.fit(X, y)
y_pred = clf_dt.predict(X)

acc = cross_val_score(estimator = clf_dt, X = X, y = y, cv = cv, scoring='f1')
acc.mean(), acc.std()

print(classification_report(y, y_pred))



''' 7: AdaBoost '''
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, classification_report
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
clf_ada = AdaBoostClassifier(random_state = 1337)
clf_ada.fit(X, y)


cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1337)
acc = cross_val_score(estimator = clf_ada, X = X, y = y, cv = cv, scoring='f1')
acc.mean(), acc.std()


parameters = {'algorithm':['SAMME', 'SAMME.R'],
              'n_estimators':[50,100,200],
              'learning_rate':[0.1,0.5,1,2,5]}
                                       
grid_search = GridSearchCV(estimator = clf_ada,
                           param_grid = parameters,
                           scoring='f1',
                           cv = cv,
                           n_jobs = -1)
start_time = time.time()
grid_search = grid_search.fit(X, y)
print('Training time: {} minutes'.format(round((time.time() - start_time)/60, 2)))
grid_search.best_params_, grid_search.best_score_


# last step
clf_ada = AdaBoostClassifier(algorithm = 'SAMME.R',
                             n_estimators = 50, 
                             learning_rate = 1,
                             random_state= 1337 )
clf_ada.fit(X, y)
y_pred = clf_ada.predict(X)

acc = cross_val_score(estimator = clf_ada, X = X, y = y, cv = cv, scoring='f1')
acc.mean(), acc.std()



def plot_curve(clf,title):
    
    train_sizes,train_scores,test_scores = learning_curve(clf, X, y, scoring = 'f1', cv = cv, random_state = 1337)

    plt.figure()
    plt.title(title)
    
    ylim = (0.4, 1.01)
    if ylim is not None:
        plt.ylim(*ylim)
        
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                train_scores_mean + train_scores_std, alpha=0.1,
                color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
        label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
        label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()

plot_curve(clf_dt,'Learning Curve of Decision Tree')
plot_curve(clf_ada,'Learning Curve of AdaBoost')
