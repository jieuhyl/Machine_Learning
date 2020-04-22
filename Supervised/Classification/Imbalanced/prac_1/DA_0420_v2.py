# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 17:31:19 2020

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


''' 4: Support Vector Machine'''
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, classification_report
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, GridSearchCV
from sklearn.svm import SVC
clf_svc = SVC(random_state = 1337)
clf_svc.fit(X, y)

# Predicting the train set results
y_pred = clf_svc.predict(X)
roc_auc_score(y, y_pred)

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1337)
acc = cross_val_score(estimator = clf_svc, X = X, y = y, cv = cv, scoring='f1')
acc.mean(), acc.std()


# KF n GS
parameters = [{'kernel': ['linear'], 
               'C': [1, 10, 20, 50, 100], 
               'gamma': [0.001, 0.01, 0.1, 1, 'auto', 'scale'],
               'class_weight':[{0:3,1:1}, {0:2,1:1}, {0:1,1:1}, {0:1,1:2}, {0:1,1:3}, 'balanced']},
    
              {'kernel': ['poly'], 
               'C': [1, 10, 20, 50, 100], 
               'gamma': [0.001, 0.01, 0.1, 1, 'auto', 'scale'], 
               'degree': [1,2,3],
               'class_weight':[{0:3,1:1}, {0:2,1:1}, {0:1,1:1}, {0:1,1:2}, {0:1,1:3}, 'balanced']},
               
              {'kernel': ['rbf'], 
               'C': [1, 10, 20, 50, 100], 
               'gamma': [0.001, 0.01, 0.1, 1, 'auto', 'scale'],
               'class_weight':[{0:3,1:1}, {0:2,1:1}, {0:1,1:1}, {0:1,1:2}, {0:1,1:3}, 'balanced']}]

grid_search = GridSearchCV(estimator = clf_svc,
                           param_grid = parameters,
                           scoring='f1',
                           cv = cv,
                           n_jobs = -1)


grid_search = grid_search.fit(X, y)
start_time = time.time()
print('Training time: {} minutes'.format(round((time.time() - start_time)/60, 2)))
grid_search.best_params_, grid_search.best_score_

# last step
clf_svc = SVC(kernel = 'rbf',
              C = 100,
              gamma = 0.01, 
              class_weight = {0: 1, 1: 2},
              probability = True,
              random_state = 1337)
clf_svc.fit(X, y)
y_pred = clf_svc.predict(X)
roc_auc_score(y, y_pred)

acc = cross_val_score(estimator = clf_svc, X = X, y = y, cv = cv, scoring='f1')
acc.mean(), acc.std()

print(classification_report(y, y_pred))


def f_importances(coef, names):
    imp = coef
    imp,names = zip(*sorted(zip(imp,names)))
    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)
    plt.show()


clf_svc = SVC(kernel = 'linear',
              C = 1,
              random_state = 1337)
clf_svc.fit(X, y)
f_importances(clf_svc.coef_[0], df.iloc[:, 2:].columns.tolist())


# calibration ================================================================
from sklearn.calibration import CalibratedClassifierCV
# wrap the model
calibrated = CalibratedClassifierCV(clf_svc)
# define grid
param_grid = dict(cv=[3,5,10], method=['sigmoid','isotonic'])
# define evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# define grid search
grid = GridSearchCV(estimator=calibrated, 
                    param_grid=param_grid, 
                    scoring='f1',
                    cv=cv, 
                    n_jobs=-1)
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
    
# calibrated model
clf_svc_cali = CalibratedClassifierCV(clf_svc, method='isotonic', cv=10) 
clf_svc_cali.fit(X, y)    
acc = cross_val_score(estimator = clf_svc_cali, X = X, y = y, cv = cv, scoring='f1')
acc.mean(), acc.std()

y_pred = clf_svc_cali.predict(X)
print(classification_report(y, y_pred))


# calibration curve ===========================================================
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split
# predict uncalibrated probabilities
def uncalibrated(trainX, testX, trainy):
	# fit a model
	model = SVC(kernel = 'rbf',
              C = 100,
              gamma = 0.01, 
              class_weight = {0: 1, 1: 2},
              probability = True,
              random_state = 1337)
	model.fit(trainX, trainy)
	# predict probabilities
	return model.predict_proba(testX)[:, 1]

# predict calibrated probabilities
def calibrated(trainX, testX, trainy):
	# define model
	model = SVC(kernel = 'rbf',
              C = 100,
              gamma = 0.01, 
              class_weight = {0: 1, 1: 2},
              probability = True,
              random_state = 1337)
	# define and fit calibration model isotonic or sigmoid
	calibrated = CalibratedClassifierCV(model, method='isotonic', cv=10)
	calibrated.fit(trainX, trainy)
	# predict probabilities
	return calibrated.predict_proba(testX)[:, 1]

# split into train/test sets
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=1337)
# uncalibrated predictions
yhat_uncalibrated = uncalibrated(trainX, testX, trainy)
# calibrated predictions
yhat_calibrated = calibrated(trainX, testX, trainy)
# reliability diagrams
#fop_uncalibrated, mpv_uncalibrated = calibration_curve(testy, yhat_uncalibrated, n_bins=10, normalize=True)
#fop_calibrated, mpv_calibrated = calibration_curve(testy, yhat_calibrated, n_bins=10)
fop_uncalibrated, mpv_uncalibrated = calibration_curve(testy, yhat_uncalibrated, n_bins=10, normalize=True)
fop_calibrated, mpv_calibrated = calibration_curve(testy, yhat_calibrated, n_bins=10)
# plot perfectly calibrated
plt.plot([0, 1], [0, 1], linestyle='--', color='black')
# plot model reliabilities
plt.plot(mpv_uncalibrated, fop_uncalibrated, marker='.', label='uncalibrated')
plt.plot(mpv_calibrated, fop_calibrated, marker='.', label='calibrated')
plt.legend()
plt.show()



# ROC =========================================================================
from sklearn.metrics import roc_curve
# predict probabilities
yhat = clf_svc_cali.predict_proba(X)
# retrieve just the probabilities for the positive class
pos_probs = yhat[:, 1]
# plot no skill roc curve
plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
# calculate roc curve for model
fpr, tpr, thresholds = roc_curve(y, pos_probs)
# plot model roc curve
plt.plot(fpr, tpr, marker='.', label='SVC')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot
plt.show()

from sklearn.metrics import roc_auc_score
# calculate roc auc
roc_auc = roc_auc_score(y, pos_probs)
print('SVC ROC AUC %.3f' % roc_auc)

# calculate the g-mean for each threshold
gmeans = np.sqrt(tpr * (1-fpr))
# locate the index of the largest g-mean
ix = np.argmax(gmeans)
print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))
# plot the roc curve for the model
plt.plot([0,1], [0,1], linestyle='--', label='No Skill')
plt.plot(fpr, tpr, marker='.', label='SVC')
plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
# show the plot
plt.show()



# PR ==========================================================================
from sklearn.metrics import precision_recall_curve
# predict probabilities
yhat = clf_svc_cali.predict_proba(X)
# retrieve just the probabilities for the positive class
pos_probs = yhat[:, 1]
# calculate the no skill line as the proportion of the positive class
no_skill = len(y[y==1]) / len(y)
# plot the no skill precision-recall curve
plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
# calculate model precision-recall curve
precision, recall, thresholds = precision_recall_curve(y, pos_probs)
# plot the model precision-recall curve
plt.plot(recall, precision, marker='.', label='SVC')
# axis labels
plt.xlabel('Recall')
plt.ylabel('Precision')
# show the legend
plt.legend()
# show the plot
plt.show()


from sklearn.metrics import auc
# calculate the precision-recall auc
auc_score = auc(recall, precision)
print('SVC PR AUC: %.3f' % auc_score)


# convert to f score
fscore = (2 * precision * recall) / (precision + recall)
# locate the index of the largest f score
ix = np.argmax(fscore)
print('Best Threshold=%f, F-Score=%.3f' % (thresholds[ix], fscore[ix]))
# plot the roc curve for the model
no_skill = len(y[y==1]) / len(y)
plt.plot([0,1], [no_skill,no_skill], linestyle='--', label='No Skill')
plt.plot(recall, precision, marker='.', label='SVC')
plt.scatter(recall[ix], precision[ix], marker='o', color='black', label='Best')
# axis labels
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
# show the plot
plt.show()
