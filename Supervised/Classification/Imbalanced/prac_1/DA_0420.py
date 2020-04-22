# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 16:27:11 2020

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


''' 1: Logistic Regression_v1'''
from sklearn.feature_selection import RFE, RFECV
from sklearn.linear_model import LogisticRegression
mod_lr = LogisticRegression()
mod_lr.fit(X, y)
y_pred = mod_lr.predict(X)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
from sklearn import  metrics
confusion_matrix(y, y_pred)
metrics.accuracy_score(y, y_pred)


rfe = RFECV(mod_lr, min_features_to_select= 10, step = 1, cv = 5)
#rfe = RFE(mod_lr, 10, step = 1)
fit = rfe.fit(X, y)

print("Num Attribute: %d" % fit.n_features_)
print("Selected Attribute: %s" % fit.support_)
print("Feature Ranking: %s" % fit.ranking_)

feature_imp = pd.DataFrame({'Attribute': df.iloc[:, 2:].columns.tolist(),
                            'Select': fit.support_,
                            'Rank':fit.ranking_}).sort_values(by='Rank', ascending=True)


pickup = feature_imp[feature_imp['Rank']==1]['Attribute']
#pickup  = df_us.iloc[:, np.r_[3:15, 39:57, 94:187]].columns[fit.support_]
   

X = df.loc[:, pickup].values

mod_lr = LogisticRegression()
mod_lr.fit(X, y)
y_pred = mod_lr.predict(X)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
from sklearn import  metrics
confusion_matrix(y, y_pred)
metrics.accuracy_score(y, y_pred)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = mod_lr, X = X, y = y, cv = 5)
accuracies.mean(), accuracies.std() 


sub = pd.DataFrame({"Attribute": df.loc[:, pickup].columns, 
                    "Coefficient": mod_lr.coef_[0,:],
                    "Impact Index": (np.exp(mod_lr.coef_[0,:])/(1+np.exp(mod_lr.coef_[0,:]))*2)*100}).sort_values(by='Impact Index', ascending=False)

sub.to_csv('C:/Users/Jie.Hu/Desktop/Driver Analysis/0331/DA_outputs_genre1_v2.csv', index=False)




''' 1: Logistic Regression_v2'''
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, classification_report
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression


clf_lr = LogisticRegression(random_state = 1337)
clf_lr.fit(X, y)

# Predicting the train set results
y_pred = clf_lr.predict(X)
roc_auc_score(y, y_pred)

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1337)
acc = cross_val_score(estimator = clf_lr, X = X, y = y, cv = cv, scoring='f1')
acc.mean(), acc.std()


# KF n GS
parameters = [{'penalty': ['l2'],
               'solver': ['newton-cg', 'lbfgs', 'sag'],
               'C': [0.01, 0.1, 1, 10, 100],
               'class_weight':[{0:3,1:1}, {0:2,1:1}, {0:1,1:1}, {0:1,1:2}, {0:1,1:3}, 'balanced']},
    
              {'penalty': ['l1','l2'],
               'solver': ['liblinear','saga'],
               'C': [0.01, 0.1, 1, 10, 100],
               'class_weight':[{0:3,1:1}, {0:2,1:1}, {0:1,1:1}, {0:1,1:2}, {0:1,1:3}, 'balanced']},
           
              {'penalty': ['elasticnet'],
               'solver': ['saga'],
               'C': [0.01, 0.1, 1, 10, 100],
               'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
               'class_weight':[{0:3,1:1}, {0:2,1:1}, {0:1,1:1}, {0:1,1:2}, {0:1,1:3}, 'balanced']}]
    

grid_search = GridSearchCV(estimator = clf_lr,
                           param_grid = parameters,
                           scoring='f1', #roc_auc, f1
                           cv = cv,
                           n_jobs = -1)

start_time = time.time()
grid_search = grid_search.fit(X, y)
print('Training time: {} minutes'.format(round((time.time() - start_time)/60, 2)))
print("Best: %f using %s" % (grid_search.best_score_, grid_search.best_params_))



# last step
clf_lr = LogisticRegression(penalty = 'elasticnet', 
                            solver = 'saga',
                            C = 0.01,
                            l1_ratio = 0.1, 
                            class_weight = {0: 1, 1: 2},
                            random_state = 1337)
clf_lr.fit(X, y)
y_pred = clf_lr.predict(X)

acc = cross_val_score(estimator = clf_lr, X = X, y = y, cv = cv, scoring='f1')
acc.mean(), acc.std()

print(classification_report(y, y_pred))
accuracy_score(y, y_pred)


sub = pd.DataFrame({"Attribute": df.iloc[:, 2:].columns, 
                    "Coefficient": clf_lr.coef_[0,:],
                    "Impact Index": (np.exp(clf_lr.coef_[0,:])/(1+np.exp(clf_lr.coef_[0,:]))*2)*100}).sort_values(by='Impact Index', ascending=False)

sub.to_csv('C:/Users/Jie.Hu/Desktop/Driver Analysis/0420/DA_outputs_lr.csv', index=False)

# ROC =========================================================================
from sklearn.metrics import roc_curve
# predict probabilities
yhat = clf_lr.predict_proba(X)
# retrieve just the probabilities for the positive class
pos_probs = yhat[:, 1]
# plot no skill roc curve
plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
# calculate roc curve for model
fpr, tpr, thresholds = roc_curve(y, pos_probs)
# plot model roc curve
plt.plot(fpr, tpr, marker='.', label='Logistic')
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
print('Logistic ROC AUC %.3f' % roc_auc)

# calculate the g-mean for each threshold
gmeans = np.sqrt(tpr * (1-fpr))
# locate the index of the largest g-mean
ix = np.argmax(gmeans)
print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))
# plot the roc curve for the model
plt.plot([0,1], [0,1], linestyle='--', label='No Skill')
plt.plot(fpr, tpr, marker='.', label='Logistic')
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
yhat = clf_lr.predict_proba(X)
# retrieve just the probabilities for the positive class
pos_probs = yhat[:, 1]
# calculate the no skill line as the proportion of the positive class
no_skill = len(y[y==1]) / len(y)
# plot the no skill precision-recall curve
plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
# calculate model precision-recall curve
precision, recall, thresholds = precision_recall_curve(y, pos_probs)
# plot the model precision-recall curve
plt.plot(recall, precision, marker='.', label='Logistic')
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
print('Logistic PR AUC: %.3f' % auc_score)

# convert to f score
fscore = (2 * precision * recall) / (precision + recall)
# locate the index of the largest f score
ix = np.argmax(fscore)
print('Best Threshold=%f, F-Score=%.3f' % (thresholds[ix], fscore[ix]))
# plot the roc curve for the model
no_skill = len(y[y==1]) / len(y)
plt.plot([0,1], [no_skill,no_skill], linestyle='--', label='No Skill')
plt.plot(recall, precision, marker='.', label='Logistic')
plt.scatter(recall[ix], precision[ix], marker='o', color='black', label='Best')
# axis labels
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
# show the plot
plt.show()