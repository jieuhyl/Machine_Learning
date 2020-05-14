# -*- coding: utf-8 -*-
"""
Created on Sun May  3 03:14:13 2020

@author: Jie.Hu
"""


''' 9: Hist Gradient Boosting'''
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
