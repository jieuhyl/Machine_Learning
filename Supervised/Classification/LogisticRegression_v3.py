# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 22:45:16 2020

@author: Jie.Hu
"""


# https://machinelearningmastery.com/cost-sensitive-logistic-regression/

# fit a logistic regression model on an imbalanced classification dataset
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
# generate dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
	n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=2)
# define model
model = LogisticRegression(solver='lbfgs')
# define evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
# summarize performance
print('Mean ROC AUC: %.3f' % mean(scores))



# weighted logistic regression model on an imbalanced classification dataset
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
# generate dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
	n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=2)
# define model
weights = {0:0.01, 1:1.0}
model = LogisticRegression(solver='lbfgs', class_weight=weights)
# define evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
# summarize performance
print('Mean ROC AUC: %.3f' % mean(scores))



# weighted logistic regression for class imbalance with heuristic weights
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
# generate dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
	n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=2)
# define model
model = LogisticRegression(solver='lbfgs', class_weight='balanced')
# define evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
# summarize performance
print('Mean ROC AUC: %.3f' % mean(scores))



# grid search class weights with logistic regression for imbalance classification
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
# generate dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
	n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=2)
# define model
model = LogisticRegression(solver='lbfgs')
# define grid
balance = [{0:100,1:1}, {0:10,1:1}, {0:1,1:1}, {0:1,1:10}, {0:1,1:100}]
param_grid = dict(class_weight=balance)
# define evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# define grid search
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=cv, scoring='roc_auc')
# execute the grid search
grid_result = grid.fit(X, y)
# report the best configuration
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# report all configurations
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))



# =============================================================================
# f1
from numpy import argmax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, classification_report
from matplotlib import pyplot    
clf_lr = LogisticRegression(penalty = 'elasticnet', solver = 'saga', l1_ratio = 0.5, random_state = 1337)

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

# KF n GS
parameters = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
              'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9], 
              'class_weight':[{0:100,1:1}, {0:10,1:1}, {0:1,1:1}, {0:1,1:10}, {0:1,1:100}, 'balanced']}

grid_search = GridSearchCV(estimator = clf_lr,
                           param_grid = parameters,
                           scoring='f1', #roc_auc, f1
                           cv = cv,
                           n_jobs = -1)

grid_search = grid_search.fit(X, y)
grid_search.best_params_, grid_search.best_score_



# last step
clf_lr = LogisticRegression(penalty = 'elasticnet', 
                            solver = 'saga',
                            C = 0.01,
                            l1_ratio = 0.7, 
                            class_weight = {0:1, 1:10},
                            random_state = 1337)
clf_lr.fit(X, y)

# predict probabilities
y_pred = clf_lr.predict_proba(X)
# keep probabilities for the positive outcome only
yhat = y_pred[:, 1]
# calculate roc curves
precision, recall, thresholds = precision_recall_curve(y, yhat)
# convert to f score
fscore = (2 * precision * recall) / (precision + recall)
# locate the index of the largest f score
ix = argmax(fscore)
print('Best Threshold=%f, F-Score=%.3f' % (thresholds[ix], fscore[ix]))
# plot the roc curve for the model
no_skill = len(y[y==1]) / len(y)
pyplot.plot([0,1], [no_skill,no_skill], linestyle='--', label='No Skill')
pyplot.plot(recall, precision, marker='.', label='Logistic')
pyplot.scatter(recall[ix], precision[ix], marker='o', color='black', label='Best')
# axis labels
pyplot.xlabel('Recall')
pyplot.ylabel('Precision')
pyplot.legend()
# show the plot
pyplot.show()


# predict by threshold
BT = thresholds[ix]
y_pred_BT = (clf_lr.predict_proba(X) >= BT).astype(int)
y_pred_BT = y_pred_BT[:, 1]

y_pred_no = clf_lr.predict(X)

print(classification_report(y, y_pred_no))
print(classification_report(y, y_pred_BT))





# =============================================================================
# ROC
from numpy import argmax, sqrt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, classification_report
from matplotlib import pyplot    
clf_lr = LogisticRegression(penalty = 'elasticnet', solver = 'saga', l1_ratio = 0.5, random_state = 1337)

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

# KF n GS
parameters = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
              'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9], 
              'class_weight':[{0:100,1:1}, {0:10,1:1}, {0:1,1:1}, {0:1,1:10}, {0:1,1:100}, 'balanced']}

grid_search = GridSearchCV(estimator = clf_lr,
                           param_grid = parameters,
                           scoring='roc_auc', #roc_auc, f1
                           cv = cv,
                           n_jobs = -1)

grid_search = grid_search.fit(X, y)
grid_search.best_params_, grid_search.best_score_



# last step
clf_lr = LogisticRegression(penalty = 'elasticnet', 
                            solver = 'saga',
                            C = 1,
                            l1_ratio = 0.1, 
                            class_weight = {0:1, 1:10},
                            random_state = 1337)
clf_lr.fit(X, y)

# predict probabilities
y_pred = clf_lr.predict_proba(X)
# keep probabilities for the positive outcome only
yhat = y_pred[:, 1]
# calculate roc curves
fpr, tpr, thresholds = roc_curve(y, yhat)
# calculate the g-mean for each threshold
gmeans = sqrt(tpr * (1-fpr))
# locate the index of the largest g-mean
ix = argmax(gmeans)
print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))
# plot the roc curve for the model
pyplot.plot([0,1], [0,1], linestyle='--', label='No Skill')
pyplot.plot(fpr, tpr, marker='.', label='Logistic')
pyplot.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
pyplot.legend()
# show the plot
pyplot.show()


for idx, i in enumerate(thresholds):
    if i <= 0.5:
        print(gmeans[idx])
        break
    
    
# predict by threshold
BT = thresholds[ix]
y_pred_BT = (clf_lr.predict_proba(X) >= BT).astype(int)
y_pred_BT = y_pred_BT[:, 1]

y_pred_no = clf_lr.predict(X)

print(classification_report(y, y_pred_no))
print(classification_report(y, y_pred_BT))



    





















