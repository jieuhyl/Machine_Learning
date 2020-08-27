# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 01:44:40 2020

@author: Jie.Hu
"""




import warnings
warnings.filterwarnings('ignore') 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, GridSearchCV
import time



# read data
df = pd.read_csv('DA_0824.csv', skipinitialspace=True, encoding='cp1252')

# check missing again
df.isnull().values.sum()
df.isnull().sum()/df.shape[0]
missing_ratio = df.isnull().sum() / len(df)

df = df.dropna()


# clean data
df.columns.tolist()
df['QMOBILEVIDTYPEr3'].value_counts()
df['QMOBILEVIDTYPEr3'] = np.where(df['QMOBILEVIDTYPEr3'] <=2, 1, 0)

'''
def mapping(x):
    if x==1:
        val = 1
    else:
        val = 0
    return val
def mapping2(x):
    return 5-x
   
df.iloc[:,2:27] = df.iloc[:,2:27].applymap(mapping2)
'''


''' model '''
# get mtx
X = df.iloc[:, 2:27].values
y = df.iloc[:,-1].values





'''  random forest '''
from sklearn.ensemble import RandomForestClassifier
clf_rf = RandomForestClassifier(random_state=1337)
clf_rf.fit(X, y)

y_pred = clf_rf.predict(X)
accuracy_score(y, y_pred)
confusion_matrix(y, y_pred)



# k fold and grid
parameters = {
              'n_estimators':[30,50,100,200],
              'max_depth':[3,5,7],
              'max_features':['auto','log2', 'sqrt']
              }
                                       
grid_search = GridSearchCV(estimator = clf_rf,
                           param_grid = parameters,
                           scoring= 'accuracy',
                           cv = 3,
                           n_jobs = -1)
start_time = time.time()
grid_search = grid_search.fit(X, y)
print('Training time: {} minutes'.format(round((time.time() - start_time)/60, 2)))
grid_search.best_params_, grid_search.best_score_


# last step
clf_rf = RandomForestClassifier(n_estimators=200,
                                max_depth=5,
                                max_features='auto',
                                random_state=1337)
clf_rf.fit(X, y)
y_pred = clf_rf.predict(X)

accuracy_score(y, y_pred)


acc = cross_val_score(estimator = clf_rf, X = X, y = y, cv = 3, scoring='accuracy')
acc.mean(), acc.std()

print(classification_report(y, y_pred))


# feature importance
plt.figure(figsize=(12, 9))
fi = clf_rf.feature_importances_
predictors = [x for x in df.iloc[:,2:27].columns]
feat_imp = pd.Series(clf_rf.feature_importances_, predictors).sort_values(ascending=False)
feat_imp.plot(kind='bar', title='Feature Importances')
plt.ylabel('Feature Importance Score')
plt.savefig('rf.png')


sub = pd.DataFrame({"Attribute": df.iloc[:, 2:27].columns, 
                    "Coefficient": clf_rf.feature_importances_,
                    "Impact Index": clf_rf.feature_importances_/(np.mean(clf_rf.feature_importances_))*100}).sort_values(by='Impact Index', ascending=False)
sub.to_csv('C:/Users/Jie.Hu/Desktop/Driver Analysis/0824/DA_outputs_rf.csv', index=False)




''' RFE '''
from sklearn.feature_selection import RFE, RFECV
rfe = RFECV(clf_rf, min_features_to_select= 10, step = 1, cv = 3)
#rfe = RFE(clf_rf , 10, step = 1)
fit = rfe.fit(X, y)

print("Num Features: %d" % fit.n_features_)
print("Selected Features: %s" % fit.support_)
print("Feature Ranking: %s" % fit.ranking_)


feature_imp = pd.DataFrame({'Features': df.iloc[:, 2:27].columns.tolist(),
                            'Select': fit.support_,
                            'Rank':fit.ranking_}).sort_values(by='Rank', ascending=True)

# merge to get label
df_label = pd.read_csv('label.csv')
sub = pd.merge(feature_imp, df_label, left_on=['Features'], right_on=['Attribute'], how='left')
sub.to_csv('C:/Users/Jie.Hu/Desktop/Driver Analysis/0330/DA_feature_imp_v1.csv', index=False)


pickup = feature_imp[feature_imp['Rank']==1]['Features']

'''
# drop
drops = ['PRG_QRACE_2', 'PRG_QRACE_4']
pickup = list(set(pickup) - set(drops))
# add new
pickup = list(set(pickup))
pickup.extend(['QRELIGIOUSLEVEL_1.0',
 'QRELIGIOUSLEVEL_2.0',
 'QRELIGIOUSLEVEL_3.0',
 'QRELIGIOUSLEVEL_4.0',
 'QRELIGIOUSLEVEL_5.0',
 'QORTHODOXFAMILIARITY_1',
 'QORTHODOXFAMILIARITY_2',
 'QORTHODOXFAMILIARITY_3',
 'QORTHODOXFAMILIARITY_4',
 'QORTHODOXFAMILIARITY_5'])
pickup = list(set(pickup)) 
'''


X_reduced = df.loc[:, pickup].values

# last step
clf_rf = RandomForestClassifier(n_estimators=200,
                                max_depth=5,
                                max_features='auto',
                                random_state=1337)
clf_rf.fit(X_reduced, y)
y_pred_reduced = clf_rf.predict(X_reduced)

accuracy_score(y, y_pred_reduced)


acc = cross_val_score(estimator = clf_rf, X = X_reduced, y = y, cv = 3, scoring='accuracy')
acc.mean(), acc.std()

print(classification_report(y, y_pred_reduced))


# feature importance
plt.figure(figsize=(12, 9))
fi = clf_rf.feature_importances_
predictors = [x for x in df.loc[:, pickup].columns]
feat_imp = pd.Series(clf_rf.feature_importances_, predictors).sort_values(ascending=False)
feat_imp.plot(kind='bar', title='Feature Importances')
plt.ylabel('Feature Importance Score')
plt.savefig('rf.png')



sub = pd.DataFrame({"Attribute": df.loc[:, pickup].columns, 
                    "Coefficient": clf_rf.feature_importances_,
                    "Impact Index": clf_rf.feature_importances_/(np.mean(clf_rf.feature_importances_))*100}).sort_values(by='Impact Index', ascending=False)

# merge to get label
df_label = pd.read_csv('label.csv')
sub = pd.merge(sub, df_label, left_on=['Attribute'], right_on=['Attribute'], how='left')
sub.to_csv('C:/Users/Jie.Hu/Desktop/Driver Analysis/0330/DA_outputs_v1.csv', index=False)