# -*- coding: utf-8 -*-
"""
Created on Fri May  1 16:33:49 2020

@author: Jie.Hu
"""



import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, classification_report, make_scorer
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, GridSearchCV, RandomizedSearchCV, train_test_split, KFold, StratifiedKFold
from scipy.stats import uniform
from scipy.stats import randint as sp_randint
import time

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




''' 4: Support Vector Machine'''
from sklearn.svm import SVC
clf_svc = SVC(probability=True, random_state = 1337)


cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1337)
acc = cross_val_score(estimator = clf_svc, X = X_train, y = y_train, cv = cv, scoring='roc_auc')
acc.mean(), acc.std()


# KF n GS
'''
parameters = [{'kernel': ['linear'], 
               'C': [1, 10, 50, 100], 
               'gamma': [0.001, 0.01, 0.1, 1, 'auto', 'scale'],
               'class_weight':[{0:3,1:1}, {0:2,1:1}, {0:1,1:1}, {0:1,1:2}, {0:1,1:3}, 'balanced']},
    
              {'kernel': ['poly'], 
               'C': [10, 50, 100, 500,1000], 
               'gamma': [0.001, 0.01, 0.1, 1, 'auto', 'scale'], 
               'degree': [1,2,3],
               'coef0': [0,1,2,3],
               'class_weight':[{0:3,1:1}, {0:2,1:1}, {0:1,1:1}, {0:1,1:2}, {0:1,1:3}, 'balanced']},
               
              {'kernel': ['rbf'], 
               'C': [1000,2000,4000], 
               'gamma': [0.001, 0.01, 0.1, 1, 'auto', 'scale'],
               'class_weight':[{0:3,1:1}, {0:2,1:1}, {0:1,1:1}, {0:1,1:2}, {0:1,1:3}, 'balanced']}]
'''

parameters = {'kernel': ['rbf'], 
               'C': [0.1,0.5,1,5,10,20], 
               'gamma': [0.0001, 0.0005, 0.001, 0.005,  0.01],
               'class_weight':[{0:2,1:1}, {0:1,1:1}, {0:1,1:2}]}

grid_search = GridSearchCV(estimator = clf_svc,
                           param_grid = parameters,
                           scoring='roc_auc',
                           cv = 3,
                           verbose = 10,
                           n_jobs = -1)

start_time = time.time()
grid_search = grid_search.fit(X_train, y_train)
print('Training time: {} minutes'.format(round((time.time() - start_time)/60, 2)))
grid_search.best_params_, grid_search.best_score_


# last step
clf_svc = grid_search.best_estimator_
#clf_svc.fit(X_train, y_train)

y_pred = clf_svc.predict(X_test)
print(classification_report(y_test, y_pred))

y_pred = clf_svc.predict_proba(X_test)[:, 1]
print('SVC AUC_ROC: %.3f' % roc_auc_score(y_test, y_pred))



''' 5: Decision Tree'''
from sklearn.tree import DecisionTreeClassifier
clf_dt = DecisionTreeClassifier(random_state = 1337)

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1337)
acc = cross_val_score(estimator = clf_dt, X = X_train, y = y_train, cv = cv, scoring='roc_auc')
acc.mean(), acc.std()


parameters = {'criterion':['gini', 'entropy'],
              'max_depth':[4,6,8],
              'max_features':['auto', 'sqrt', 'log2'],
              'min_samples_leaf':[1,3,5],
              'min_samples_split':[2,4,6],
              'class_weight':[{0:3,1:1}, {0:2,1:1}, {0:1,1:1}, {0:1,1:2}, {0:1,1:3}, 'balanced'],
              'ccp_alpha':[0.00001, 0.00005, 0.0001, 0.0005, 0.001]}
                                       
grid_search = GridSearchCV(estimator = clf_dt,
                           param_grid = parameters,
                           scoring='roc_auc',
                           cv = 3,
                           verbose = 10,
                           n_jobs = -1)
start_time = time.time()
grid_search = grid_search.fit(X_train, y_train)
print('Training time: {} minutes'.format(round((time.time() - start_time)/60, 2)))
grid_search.best_params_, grid_search.best_score_


# last step
clf_dt = grid_search.best_estimator_
clf_dt.fit(X_train, y_train)

y_pred = clf_dt.predict(X_test)
print(classification_report(y_test, y_pred))

y_pred = clf_dt.predict_proba(X_test)[:, 1]
print('DT AUC_ROC: %.3f' % roc_auc_score(y_test, y_pred))



''' 6: Random Forest'''
from sklearn.ensemble import RandomForestClassifier
clf_rf = RandomForestClassifier(random_state = 1337)


cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1337)
acc = cross_val_score(estimator = clf_rf, X = X_train, y = y_train, cv = cv, scoring='roc_auc')
acc.mean(), acc.std()


# KF n GS
parameters = {'criterion':['gini'],
              'max_depth':[7,9,11],
              #'max_features':['auto', 'sqrt', 'log2'],
              'max_features':['auto'],
              'max_samples':[0.8,0.9],
              'min_samples_leaf':[1,3],
              'min_samples_split':[2,4],
              #'class_weight':['balanced', 'balanced_subsample', None],
              'n_estimators':[400,500,600],
              'ccp_alpha':[0.0001,0.001,0.01],
              'bootstrap':[True],
              'oob_score':[True]}
                                       
grid_search = GridSearchCV(estimator = clf_rf,
                           param_grid = parameters,
                           scoring='roc_auc',
                           cv = 3,
                           verbose= 10,
                           n_jobs = -1)

start_time = time.time()
grid_search = grid_search.fit(X_train, y_train)
print('Training time: {} minutes'.format(round((time.time() - start_time)/60, 2)))
grid_search.best_params_, grid_search.best_score_


# last step
clf_rf = grid_search.best_estimator_
clf_rf.fit(X_train, y_train)

y_pred = clf_rf.predict(X_test)
print(classification_report(y_test, y_pred))

y_pred = clf_rf.predict_proba(X_test)[:, 1]
print('RF AUC_ROC: %.3f' % roc_auc_score(y_test, y_pred))



''' 7: Gradient Boosting'''
from sklearn.ensemble import GradientBoostingClassifier
clf_gb = GradientBoostingClassifier(validation_fraction=0.2,
                                    n_iter_no_change=20, 
                                    tol=0.001,
                                    random_state= 1337)

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1337)
acc = cross_val_score(estimator = clf_gb, X = X_train, y = y_train, cv = cv, scoring='roc_auc')
acc.mean(), acc.std()

# KF & GS
#gmean_scorer = make_scorer(geometric_mean_score, greater_is_better=True)

parameters = {'learning_rate':[0.01, 0.1, 0.2], 
              'max_depth':[4,6,8],
              'max_features':['auto', 'sqrt', 'log2'],
              'min_samples_leaf':[1,2,3],
              'min_samples_split':[2,4,6],
              'n_estimators':[100,200,400],
              'subsample':[0.5,0.7,0.9]}

                                       
grid_search = GridSearchCV(estimator = clf_gb,
                           param_grid = parameters,
                           scoring='roc_auc',
                           cv = 3,
                           verbose = 10,
                           n_jobs = -1)

start_time = time.time()
grid_search = grid_search.fit(X_train, y_train)
print('Training time: {} minutes'.format(round((time.time() - start_time)/60, 2)))
grid_search.best_params_, grid_search.best_score_


# last step
clf_gb = grid_search.best_estimator_
clf_gb.fit(X_train, y_train)

y_pred = clf_gb.predict(X_test)
print(classification_report(y_test, y_pred))

y_pred = clf_gb.predict_proba(X_test)[:, 1]
print('GB AUC_ROC: %.3f' % roc_auc_score(y_test, y_pred))




''' 8: HistGradient Boosting'''
from sklearn.experimental import enable_hist_gradient_boosting 
from sklearn.ensemble import HistGradientBoostingClassifier
clf_hgb = HistGradientBoostingClassifier(validation_fraction=0.2,
                                         n_iter_no_change=20, 
                                         tol=0.001,
                                         random_state= 1337)

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1337)
acc = cross_val_score(estimator = clf_gb, X = X_train, y = y_train, cv = cv, scoring='roc_auc')
acc.mean(), acc.std()

# KF & GS
#gmean_scorer = make_scorer(geometric_mean_score, greater_is_better=True)

parameters = {'learning_rate':[0.001, 0.01, 0.1], 
              'max_depth':[3,5,7],
              'max_leaf_nodes':[11,21,31],
              'min_samples_leaf':[1,3,5],
              'max_iter':[100,200,400],
              'l2_regularization':[0.001,0.01,0.1]}

                                       
grid_search = GridSearchCV(estimator = clf_hgb,
                           param_grid = parameters,
                           scoring='roc_auc',
                           cv = 3,
                           verbose = 10,
                           n_jobs = -1)

start_time = time.time()
grid_search = grid_search.fit(X_train, y_train)
print('Training time: {} minutes'.format(round((time.time() - start_time)/60, 2)))
grid_search.best_params_, grid_search.best_score_


# last step
clf_hgb = grid_search.best_estimator_
clf_hgb.fit(X_train, y_train)

y_pred = clf_hgb.predict(X_test)
print(classification_report(y_test, y_pred))

y_pred = clf_hgb.predict_proba(X_test)[:, 1]
print('HGB AUC_ROC: %.3f' % roc_auc_score(y_test, y_pred))


# KF & RS
parameters = {'learning_rate': uniform(0,0.1), 
              'max_depth':sp_randint(3, 11),
              'max_leaf_nodes':sp_randint(2, 32),
              'min_samples_leaf':sp_randint(1, 11),
              'max_iter':[400,600,800,1000,1200],
              'l2_regularization':uniform(0,0.1)}

rand_search = RandomizedSearchCV(estimator = clf_hgb,
                                 param_distributions = parameters,
                                 scoring='roc_auc',
                                 n_iter=100,
                                 cv = 3,
                                 verbose = 10,
                                 n_jobs = -1)

start_time = time.time()
rand_search = rand_search.fit(X_train, y_train)
print('Training time: {} minutes'.format(round((time.time() - start_time)/60, 2)))
rand_search.best_params_, rand_search.best_score_

# last step
clf_hgb_rand = rand_search.best_estimator_
clf_hgb_rand.fit(X_train, y_train)

y_pred = clf_hgb_rand.predict(X_test)
print(classification_report(y_test, y_pred))

y_pred = clf_hgb_rand.predict_proba(X_test)[:, 1]
print('HGB AUC_ROC: %.3f' % roc_auc_score(y_test, y_pred))



results = []
for clf in [clf_svc, clf_dt, clf_rf, clf_gb, clf_hgb, clf_hgb_rand]:
    #print(clf)
    y_pred = clf.predict(X_test)
    y_pred_prob = clf.predict_proba(X_test)[:, 1]
    results.append((accuracy_score(y_test, y_pred), 
                    f1_score(y_test, y_pred),
                    geometric_mean_score(y_test, y_pred),
                    roc_auc_score(y_test, y_pred_prob)))
                    
                    
df_results = pd.DataFrame(results)
df_results.insert(0, 'clf', ['SVC', 'DT', 'RF', 'GB', 'HGB', 'HGB_RAND'])  
df_results.rename(columns={0:'acc', 1: 'f1', 2:'gmean', 3:'roc_auc'}, inplace=True)  



    
# ensemble ====================================================================
# voting
from sklearn.ensemble import VotingClassifier

levels = list()
levels.append(('svc', clf_svc))
#models.append(('dt',  clf_dt))
levels.append(('rf',  clf_rf))
levels.append(('gb',  clf_gb))
levels.append(('hgb', clf_hgb))
# define the soft voting ensemble
clf_voting = VotingClassifier(estimators=levels, 
                              voting='soft')

# KF & GS
parameters = {'weights':[[2,1,1,1], [1,2,1,1], [1,1,2,1], [1,1,1,2], None]}

grid_search = GridSearchCV(estimator = clf_voting,
                           param_grid = parameters,
                           scoring='roc_auc',
                           cv = 3,
                           verbose = 10,
                           n_jobs = -1)

start_time = time.time()
grid_search = grid_search.fit(X_train, y_train)
print('Training time: {} minutes'.format(round((time.time() - start_time)/60, 2)))
grid_search.best_params_, grid_search.best_score_

# last step
clf_voting = grid_search.best_estimator_



    
# stacking
from sklearn.ensemble import StackingClassifier
# define the base models
level0 = list()
#level0.append(('svc', clf_svc))
level0.append(('rf',  clf_rf))
level0.append(('gb',  clf_gb))
level0.append(('hgb', clf_hgb))
# define meta learner model
#level1 = clf_svc
level1 = SVC(probability=True, random_state = 1337)
# define the stacking ensemble
clf_stacking = StackingClassifier(estimators=level0, 
                                  final_estimator=level1, 
                                  cv=5)

# KF & RS
parameters = {'final_estimator__kernel': ['rbf'],
              'final_estimator__C': [500,1000,2000,4000], 
              'final_estimator__gamma': uniform(0,0.1)}

rand_search = RandomizedSearchCV(estimator = clf_stacking,
                                 param_distributions = parameters,
                                 scoring='roc_auc',
                                 n_iter=100,
                                 cv = 3,
                                 verbose = 10,
                                 n_jobs = -1)

start_time = time.time()
rand_search = rand_search.fit(X_train, y_train)
print('Training time: {} minutes'.format(round((time.time() - start_time)/60, 2)))
rand_search.best_params_, rand_search.best_score_
# last step
clf_stacking = rand_search.best_estimator_



# multi stacking
final_layer = StackingClassifier(
            estimators=[('hgb', clf_gb),
                        ('hgb_r', clf_hgb_rand)],
            final_estimator=SVC(kernel = 'rbf', C= 2000, gamma = 0.001, probability=True, random_state = 1337))

multi_layer = StackingClassifier(
            estimators=[('rf', clf_rf),
                        ('gb', clf_gb),
                        ('hgb', clf_hgb)],
            final_estimator=final_layer,
            cv = 5)

multi_layer.fit(X_train, y_train)



# record results
results = []
for clf in [clf_svc, clf_dt, clf_rf, clf_gb, clf_hgb, clf_voting, clf_stacking, multi_layer]:
    #print(clf)
    y_pred = clf.predict(X_test)
    y_pred_prob = clf.predict_proba(X_test)[:, 1]
    results.append((accuracy_score(y_test, y_pred), 
                    f1_score(y_test, y_pred),
                    geometric_mean_score(y_test, y_pred),
                    roc_auc_score(y_test, y_pred_prob)))
                    
                    
df_results = pd.DataFrame(results)
df_results.insert(0, 'clf', ['SVC', 'DT', 'RF', 'GB', 'HGB', 'VOTING', 'STACKING', 'MULTI_STACKING'])  
df_results.rename(columns={0:'acc', 1: 'f1', 2:'gmean', 3:'roc_auc'}, inplace=True)  



#==============================================================================
# 1 skf
skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 1337)
# Create arrays and dataframes to store results
auc_preds = []
test_pred_prob = np.zeros(df_test.shape[0])
n_fold = 0
for idx_train, idx_valid in skf.split(X_train, y_train):
    train_x, train_y = X_train[idx_train], y_train[idx_train]
    valid_x, valid_y = X_train[idx_valid], y_train[idx_valid]
    
    model = clf_hgb_rand
    
    model.fit(train_x, train_y)
    
    oof_preds = model.predict_proba(valid_x)[:, 1]
    test_pred_prob += model.predict_proba(df_test[predictors])[:, 1]/5 
    
    print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds)))
    auc_preds.append(roc_auc_score(valid_y, oof_preds))
    #del model, train_x, train_y, valid_x, valid_y
    #gc.collect()
    n_fold = n_fold + 1
print('Full AUC score %.6f' % np.mean(auc_preds))   

test_pred = (test_pred_prob >= 0.5).astype(bool)  
    
print('EN_HGB ACC: %.3f' % accuracy_score(y_test, test_pred))
print('EN_HGB F1: %.3f' % f1_score(y_test, test_pred))
print('EN_HGB GMEAN: %.3f' % geometric_mean_score(y_test, test_pred))    
print('EN_HGB AUC_ROC: %.3f' % roc_auc_score(y_test, test_pred_prob))


# 2 rskf
#kf = KFold(n_splits = 5, random_state = 1337, shuffle = True)
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=1337)
# Create arrays and dataframes to store results
auc_preds = []
test_pred_prob = np.zeros(df_test.shape[0])
n_fold = 0
for idx_train, idx_valid in rskf.split(X_train, y_train):
    train_x, train_y = X_train[idx_train], y_train[idx_train]
    valid_x, valid_y = X_train[idx_valid], y_train[idx_valid]
    
    model = clf_hgb_rand
    
    model.fit(train_x, train_y)
    
    oof_preds = model.predict_proba(valid_x)[:, 1]
    test_pred_prob += model.predict_proba(df_test[predictors])[:, 1]/10 
    
    print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds)))
    auc_preds.append(roc_auc_score(valid_y, oof_preds))
    #del model, train_x, train_y, valid_x, valid_y
    #gc.collect()
    n_fold = n_fold + 1
print('Full AUC score %.6f' % np.mean(auc_preds))   

test_pred = (test_pred_prob >= 0.5).astype(bool)  
    
print('EN_HGB ACC: %.3f' % accuracy_score(y_test, test_pred))
print('EN_HGB F1: %.3f' % f1_score(y_test, test_pred))
print('EN_HGB GMEAN: %.3f' % geometric_mean_score(y_test, test_pred))    
print('EN_HGB AUC_ROC: %.3f' % roc_auc_score(y_test, test_pred_prob))




























