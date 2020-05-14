# -*- coding: utf-8 -*-
"""
Created on Sun May  3 04:45:01 2020

@author: Jie.Hu
"""



# dt =========================================================================
''' 8: Hist Gradient Boosting'''
from sklearn.experimental import enable_hist_gradient_boosting 
from sklearn.ensemble import HistGradientBoostingRegressor
mod_hgb = HistGradientBoostingRegressor(validation_fraction=0.2,
                                         n_iter_no_change=20, 
                                         tol=0.001,
                                         random_state= 1337)
mod_hgb.fit(X_train, y_train)


# Predicting the Test set results
y_pred = mod_hgb.predict(X_test)
mape = np.mean(np.abs((np.expm1(y_test) - np.expm1(y_pred)) / np.expm1(y_test))) * 100
#Print model report:
print ("\nSVM Model Report")
print ("MAPE : %.2f" % mape)


# hyperparameters tuning
def my_scorer(y_true, y_pred):
    mape = np.mean(np.abs((np.expm1(y_true) - np.expm1(y_pred)) / np.expm1(y_true))) * 100
    return mape
my_func = make_scorer(my_scorer, greater_is_better=False)


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
mod_hgb = grid_search.best_estimator_

y_pred1 = mod_dt.predict(X_train)
y_pred2 = mod_dt.predict(X_test)
score1 = np.mean(np.abs((np.expm1(y_train) - np.expm1(y_pred1)) / np.expm1(y_train))) * 100
score2 = np.mean(np.abs((np.expm1(y_test) - np.expm1(y_pred2)) / np.expm1(y_test))) * 100
print ("\nSVM Model Report")
print("train {:.2f} | valid {:.2f}".format(float(score1), float(score2)))



# KF & RS =====================================================================
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
