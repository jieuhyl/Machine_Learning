# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 02:25:06 2020

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


''' ALLKNN '''
from collections import Counter
from imblearn.under_sampling import AllKNN, NeighbourhoodCleaningRule
# define undersampling strategy
under_allknn = NeighbourhoodCleaningRule()
# fit and apply the transform
X, y = under_allknn.fit_resample(X, y)
# summarize class distribution
print(Counter(y))



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
              'min_samples_leaf':[1,3,5,7],
              'min_samples_split':[2,6,10,14],
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
                                #max_leaf_nodes = 10,
                                max_features = 'auto',
                                min_samples_leaf = 1,
                                min_samples_split = 14,
                                class_weight = {0: 1, 1: 1},
                                random_state= 1337 )
clf_dt.fit(X, y)
y_pred = clf_dt.predict(X)

acc = cross_val_score(estimator = clf_dt, X = X, y = y, cv = cv, scoring='f1')
acc.mean(), acc.std()

print(classification_report(y, y_pred))



# feature importance
fi = clf_dt.feature_importances_
predictors = [x for x in df.iloc[:, 2:].columns]
feat_imp = pd.Series(fi, predictors).sort_values(ascending=False)
feat_imp.plot(kind='bar', title='Feature Importances')
plt.ylabel('Feature Importance Score')

sub = pd.DataFrame({"Attribute": df.iloc[:, 2:].columns, 
                    "Coefficient": clf_dt.feature_importances_,
                    "Impact Index": clf_dt.feature_importances_/(np.mean(clf_dt.feature_importances_))*100}).sort_values(by='Impact Index', ascending=False)
sub.to_csv('C:/Users/Jie.Hu/Desktop/Driver Analysis/0420/DA_outputs_dt.csv', index=False)



from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus

feature_cols = df.iloc[:,2:].columns

dot_data = StringIO()

export_graphviz(clf_dt, 
                out_file=dot_data,  
                filled=True, 
                rounded=True,
                special_characters=True, 
                feature_names = feature_cols,
                class_names=['Other','Top Box'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('explore.png')
Image(graph.create_png())



''' 7: AdaBoost '''
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, classification_report
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
clf_ada = AdaBoostClassifier(clf_dt, random_state = 1337)
clf_ada.fit(X, y)


cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1337)
acc = cross_val_score(estimator = clf_ada, X = X, y = y, cv = cv, scoring='f1')
acc.mean(), acc.std()


parameters = {'algorithm':['SAMME', 'SAMME.R'],
              'n_estimators':[50,100,200,400],
              'learning_rate':[0.01,0.1,0.5,1]}
                                       
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
clf_ada = AdaBoostClassifier(clf_dt,
                             algorithm = 'SAMME.R',
                             n_estimators = 200, 
                             learning_rate = 0.01,
                             random_state= 1337 )
clf_ada.fit(X, y)
y_pred = clf_ada.predict(X)

acc = cross_val_score(estimator = clf_ada, X = X, y = y, cv = cv, scoring='f1')
acc.mean(), acc.std()

print(classification_report(y, y_pred))


# feature importance
fi = clf_ada.feature_importances_
predictors = [x for x in df.iloc[:, 2:].columns]
feat_imp = pd.Series(fi, predictors).sort_values(ascending=False)
feat_imp.plot(kind='bar', title='Feature Importances')
plt.ylabel('Feature Importance Score')

sub = pd.DataFrame({"Attribute": df.iloc[:, 2:].columns, 
                    "Coefficient": clf_ada.feature_importances_,
                    "Impact Index": clf_ada.feature_importances_/(np.mean(clf_ada.feature_importances_))*100}).sort_values(by='Impact Index', ascending=False)
sub.to_csv('C:/Users/Jie.Hu/Desktop/Driver Analysis/0420/DA_outputs_ada.csv', index=False)