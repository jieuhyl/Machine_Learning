# -*- coding: utf-8 -*-
"""
Created on Fri May  1 02:22:12 2020

@author: Jie.Hu
"""


import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
import time

''' 7: AdaBoost '''
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, classification_report
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, GridSearchCV, train_test_split, KFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, weights=[0.5], random_state=1337)

df = pd.DataFrame(X)
df['Class'] = y

target = 'Class'
predictors = df.iloc[:,0:20].columns.tolist()

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, shuffle=True, random_state=1337)
df_train, df_test = train_test_split(df, test_size=0.2, random_state=1337)


# mtx
X_train = df_train.iloc[:,0:20].values
y_train = df_train['Class'].values

X_test = df_test[predictors].values
y_test = df_test[target].values


# define the model
lr = LogisticRegression()
nb = GaussianNB()
dt = DecisionTreeClassifier(random_state = 1337)

clf_ada = AdaBoostClassifier(base_estimator=dt)
# evaluate the model
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1337)
acc = cross_val_score(estimator = clf_ada, X = X_train, y = y_train, cv = cv, scoring='f1')
acc.mean(), acc.std()


# last step
clf_ada = AdaBoostClassifier(dt,
                             algorithm = 'SAMME',
                             n_estimators = 100, 
                             learning_rate = 0.1,
                             random_state= 1337 )
clf_ada.fit(X_train, y_train)
y_pred = clf_ada.predict(X_test)
print(classification_report(y_test, y_pred))


y_pred = clf_ada.predict_proba(X_test)[:, 1]
roc_auc_score(y_test, y_pred)




#kf = KFold(n_splits = 5, random_state = 1337, shuffle = True)
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1337)
# Create arrays and dataframes to store results
auc_preds = []
test_preds = np.zeros(df_test.shape[0])
n_fold = 0
for idx_train, idx_valid in rskf.split(X_train, y_train):
    train_x, train_y = X_train[idx_train], y_train[idx_train]
    valid_x, valid_y = X_train[idx_valid], y_train[idx_valid]
    
    model = AdaBoostClassifier(dt,
                             algorithm = 'SAMME',
                             n_estimators = 100, 
                             learning_rate = 0.1,
                             random_state= 1337 )
    
    model.fit(train_x, train_y)
    
    oof_preds = model.predict_proba(valid_x)[:, 1]
    test_preds += model.predict_proba(df_test[predictors])[:, 1]/15 
    
    print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds)))
    auc_preds.append(roc_auc_score(valid_y, oof_preds))
    #del model, train_x, train_y, valid_x, valid_y
    #gc.collect()
    n_fold = n_fold + 1
print('Full AUC score %.6f' % np.mean(auc_preds))   

roc_auc_score(y_test, test_preds)





'''
kf = KFold(n_splits = NUMBER_KFOLDS, random_state = RANDOM_STATE, shuffle = True)
# Create arrays and dataframes to store results
oof_preds = np.zeros(train_df.shape[0])
test_preds = np.zeros(test_df.shape[0])
feature_importance_df = pd.DataFrame()
n_fold = 0
for train_idx, valid_idx in kf.split(train_df):
    train_x, train_y = train_df[predictors].iloc[train_idx],train_df[target].iloc[train_idx]
    valid_x, valid_y = train_df[predictors].iloc[valid_idx],train_df[target].iloc[valid_idx]
    
    evals_results = {}
    model =  LGBMClassifier(
                  nthread=-1,
                  n_estimators=2000,
                  learning_rate=0.01,
                  num_leaves=80,
                  colsample_bytree=0.98,
                  subsample=0.78,
                  reg_alpha=0.04,
                  reg_lambda=0.073,
                  subsample_for_bin=50,
                  boosting_type='gbdt',
                  is_unbalance=False,
                  min_split_gain=0.025,
                  min_child_weight=40,
                  min_child_samples=510,
                  objective='binary',
                  metric='auc',
                  silent=-1,
                  verbose=-1,
                  feval=None)
    model.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], 
                eval_metric= 'auc', verbose= VERBOSE_EVAL, early_stopping_rounds= EARLY_STOP)
    
    oof_preds[valid_idx] = model.predict_proba(valid_x, num_iteration=model.best_iteration_)[:, 1]
    test_preds += model.predict_proba(test_df[predictors], num_iteration=model.best_iteration_)[:, 1] / kf.n_splits
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = predictors
    fold_importance_df["importance"] = clf.feature_importances_
    fold_importance_df["fold"] = n_fold + 1
    
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
    del model, train_x, train_y, valid_x, valid_y
    gc.collect()
    n_fold = n_fold + 1
train_auc_score = roc_auc_score(train_df[target], oof_preds)
print('Full AUC score %.6f' % train_auc_score)  


from sklearn.model_selection import RepeatedStratifiedKFold
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([0, 0, 1, 1])

rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=4, random_state=42)
for train_index, test_index in rskf.split(X, y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]


kf = KFold(n_splits = 3, random_state = 1337, shuffle = True)
for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]



y_true = np.array([0, 0, 1, 1])
y_scores = np.array([0.01, 0.07, 0.8, 0.3])
roc_auc_score(y_true, y_scores)




















