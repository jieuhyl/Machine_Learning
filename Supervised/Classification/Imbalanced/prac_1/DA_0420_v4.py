# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 00:07:31 2020

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



''' 6: Random Forest'''
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, classification_report
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
clf_rf = RandomForestClassifier(random_state = 1337)
clf_rf.fit(X, y)


cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1337)
acc = cross_val_score(estimator = clf_rf, X = X, y = y, cv = cv, scoring='roc_auc')
acc.mean(), acc.std()


# KF n GS
parameters = {'criterion':['gini', 'entropy'],
              'max_depth':[4,5,6],
              'max_features':['auto', 'sqrt', 'log2'],
              'max_samples':[0.3,0.7],
              'min_samples_leaf':[1,2,3],
              'min_samples_split':[4,6],
              'class_weight':['balanced', 'balanced_subsample', None],
              'n_estimators':[50, 100],
              'oob_score':[True]}
                                       
grid_search = GridSearchCV(estimator = clf_rf,
                           param_grid = parameters,
                           scoring='roc_auc',
                           cv = cv,
                           n_jobs = -1)

start_time = time.time()
grid_search = grid_search.fit(X, y)
print('Training time: {} minutes'.format(round((time.time() - start_time)/60, 2)))
grid_search.best_params_, grid_search.best_score_


# last step
clf_rf = RandomForestClassifier(criterion='gini',
                                    max_depth = 4, 
                                    max_features = 'auto',
                                    max_samples = 0.3,
                                    min_samples_leaf = 2, 
                                    min_samples_split = 4,
                                    n_estimators=100,
                                    class_weight = None,
                                    oob_score = True,
                                    random_state = 1337)
clf_rf.fit(X, y)
y_pred = clf_rf.predict(X)

acc = cross_val_score(estimator = clf_rf, X = X, y = y, cv = cv, scoring='roc_auc')
acc.mean(), acc.std()

print(classification_report(y, y_pred))




# recall
# KF n GS
parameters = {'criterion':['gini'],
              'max_depth':[4,6],
              'max_features':['auto', 'sqrt', 'log2'],
              'max_samples':[0.3,0.7],
              'min_samples_leaf':[1,2,3],
              'min_samples_split':[2,4,6],
              'class_weight':['balanced', 'balanced_subsample', None],
              'n_estimators':[100],
              'oob_score':[True]}
                                       
grid_search = GridSearchCV(estimator = clf_rf,
                           param_grid = parameters,
                           scoring='f1',
                           cv = cv,
                           n_jobs = -1)

start_time = time.time()
grid_search = grid_search.fit(X, y)
print('Training time: {} minutes'.format(round((time.time() - start_time)/60, 2)))
grid_search.best_params_, grid_search.best_score_

# last step
clf_rf = RandomForestClassifier(criterion='gini',
                                    max_depth = 4, 
                                    max_features = 'auto',
                                    max_samples = 0.3,
                                    min_samples_leaf = 1, 
                                    min_samples_split = 4,
                                    n_estimators=100,
                                    class_weight = 'balanced_subsample',
                                    oob_score = True,
                                    random_state = 1337)
clf_rf.fit(X, y)
y_pred = clf_rf.predict(X)

acc = cross_val_score(estimator = clf_rf, X = X, y = y, cv = cv, scoring='f1')
acc.mean(), acc.std()

print(classification_report(y, y_pred))



# feature importance
fi = clf_rf.feature_importances_
predictors = [x for x in df.iloc[:, 2:].columns]
feat_imp = pd.Series(fi, predictors).sort_values(ascending=False)
feat_imp.plot(kind='bar', title='Feature Importances')
plt.ylabel('Feature Importance Score')

sub = pd.DataFrame({"Attribute": df.iloc[:, 2:].columns, 
                    "Coefficient": clf_rf.feature_importances_,
                    "Impact Index": clf_rf.feature_importances_/(np.mean(clf_rf.feature_importances_))*100}).sort_values(by='Impact Index', ascending=False)
sub.to_csv('C:/Users/Jie.Hu/Desktop/Driver Analysis/0420/DA_outputs_rf.csv', index=False)




''' not helpful 
# calibration ================================================================
from sklearn.calibration import CalibratedClassifierCV
# wrap the model
calibrated = CalibratedClassifierCV(clf_rf)
# define grid
param_grid = dict(cv=[3,5,10], method=['sigmoid','isotonic'])
# define evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1337)
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
clf_rf_cali = CalibratedClassifierCV(clf_rf, method='isotonic', cv=10) 
clf_rf_cali.fit(X, y)    
acc = cross_val_score(estimator = clf_rf_cali, X = X, y = y, cv = cv, scoring='f1')
acc.mean(), acc.std()



# calibration curve ===========================================================
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split
# predict uncalibrated probabilities
def uncalibrated(trainX, testX, trainy):
	# fit a model
	model = RandomForestClassifier(criterion='gini',
                                    max_depth = 4, 
                                    max_features = 'auto',
                                    max_samples = 0.3,
                                    min_samples_leaf = 1, 
                                    min_samples_split = 4,
                                    n_estimators=100,
                                    class_weight = 'balanced_subsample',
                                    oob_score = True,
                                    random_state = 1337)
	model.fit(trainX, trainy)
	# predict probabilities
	return model.predict_proba(testX)[:, 1]

# predict calibrated probabilities
def calibrated(trainX, testX, trainy):
	# define model
	model = RandomForestClassifier(criterion='gini',
                                    max_depth = 4, 
                                    max_features = 'auto',
                                    max_samples = 0.3,
                                    min_samples_leaf = 1, 
                                    min_samples_split = 4,
                                    n_estimators=100,
                                    class_weight = 'balanced_subsample',
                                    oob_score = True,
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
'''


# ROC =========================================================================
from sklearn.metrics import roc_curve
# predict probabilities
yhat = clf_rf.predict_proba(X)
# retrieve just the probabilities for the positive class
pos_probs = yhat[:, 1]
# plot no skill roc curve
plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
# calculate roc curve for model
fpr, tpr, thresholds = roc_curve(y, pos_probs)
# plot model roc curve
plt.plot(fpr, tpr, marker='.', label='RF')
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
print('RF ROC AUC %.3f' % roc_auc)

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
yhat = clf_rf.predict_proba(X)
# retrieve just the probabilities for the positive class
pos_probs = yhat[:, 1]
# calculate the no skill line as the proportion of the positive class
no_skill = len(y[y==1]) / len(y)
# plot the no skill precision-recall curve
plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
# calculate model precision-recall curve
precision, recall, thresholds = precision_recall_curve(y, pos_probs)
# plot the model precision-recall curve
plt.plot(recall, precision, marker='.', label='RF')
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
print('RF PR AUC: %.3f' % auc_score)

# convert to f score
fscore = (2 * precision * recall) / (precision + recall)
# locate the index of the largest f score
ix = np.argmax(fscore)
print('Best Threshold=%f, F-Score=%.3f' % (thresholds[ix], fscore[ix]))
# plot the roc curve for the model
no_skill = len(y[y==1]) / len(y)
plt.plot([0,1], [no_skill,no_skill], linestyle='--', label='No Skill')
plt.plot(recall, precision, marker='.', label='RF')
plt.scatter(recall[ix], precision[ix], marker='o', color='black', label='Best')
# axis labels
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
# show the plot
plt.show()