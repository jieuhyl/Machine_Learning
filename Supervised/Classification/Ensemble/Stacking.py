# -*- coding: utf-8 -*-
"""
Created on Sun May  3 04:32:06 2020

@author: Jie.Hu
"""


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