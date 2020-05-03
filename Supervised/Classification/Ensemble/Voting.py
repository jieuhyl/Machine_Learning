# -*- coding: utf-8 -*-
"""
Created on Sun May  3 04:26:22 2020

@author: Jie.Hu
"""


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
