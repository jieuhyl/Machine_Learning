# -*- coding: utf-8 -*-
"""
Created on Thu May 14 16:44:22 2020

@author: Jie.Hu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, classification_report, make_scorer
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, GridSearchCV, RandomizedSearchCV, train_test_split, KFold, StratifiedKFold
import time

# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, weights=[0.5], random_state=1337)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, shuffle=True, random_state=1337)


''' 5: Decision Tree'''
from sklearn.tree import DecisionTreeClassifier
clf_dt = DecisionTreeClassifier(random_state = 1337)

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1337)
acc = cross_val_score(estimator = clf_dt, X = X_train, y = y_train, cv = cv, scoring='roc_auc')
acc.mean(), acc.std()


parameters = {'criterion':['entropy'],
              'max_depth':[8,9,10,11,12],
              'max_features':['auto'],
              'min_samples_leaf':[7,8,9,10,11,12,13],
              'min_samples_split':[2,3,4,5]}
                                       
grid_search = GridSearchCV(estimator = clf_dt,
                           param_grid = parameters,
                           scoring='roc_auc',
                           cv = cv,
                           verbose = 100,
                           n_jobs = -1)
start_time = time.time()
grid_search = grid_search.fit(X_train, y_train)
print('Training time: {} minutes'.format(round((time.time() - start_time)/60, 2)))
grid_search.best_params_, grid_search.best_score_

# last step
clf_dt_grid = grid_search.best_estimator_

y_pred = clf_dt_grid.predict(X_test)
print(classification_report(y_test, y_pred))

y_pred = clf_dt_grid.predict_proba(X_test)[:, 1]
print('DT AUC_ROC: %.3f' % roc_auc_score(y_test, y_pred))


#===
clf = DecisionTreeClassifier(criterion='entropy',
                             max_depth= 10,
                             max_features= 'auto',
                             min_samples_leaf= 10,
                             min_samples_split= 2,
                             random_state = 1337)
path = clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities


clfs = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(ccp_alpha=ccp_alpha,
                                 criterion='entropy',
                                 max_depth= 10,
                                 max_features= 'auto',
                                 min_samples_leaf= 10,
                                 min_samples_split= 2,
                                 random_state = 1337)
    clf.fit(X_train, y_train)
    clfs.append(clf)
print("Number of nodes in the last tree is: {} with ccp_alpha: {}".format(clfs[-1].tree_.node_count, ccp_alphas[-1]))


clfs = clfs[:-1]
ccp_alphas = ccp_alphas[:-1]

train_scores = [clf.score(X_train, y_train) for clf in clfs]
test_scores = [clf.score(X_test, y_test) for clf in clfs]

fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(ccp_alphas, train_scores, marker='o', label="train",
        drawstyle="steps-post")
ax.plot(ccp_alphas, test_scores, marker='o', label="test",
        drawstyle="steps-post")
ax.legend()
plt.show()



# last step
clf_dt = DecisionTreeClassifier(ccp_alpha=0.02,
                                 criterion='entropy',
                                 max_depth= 7,
                                 max_features= 'auto',
                                 min_samples_leaf= 7,
                                 min_samples_split= 2,
                                 random_state = 1337)
clf_dt.fit(X_train, y_train)
y_pred = clf_dt.predict(X_test)


print(classification_report(y_test, y_pred))

df1 = pd.DataFrame(data=X,columns=['V'+str(i) for i in range(1,20+1)])
df2 = pd.DataFrame(data=y,columns=['Target'])
df = pd.concat([df1,df2],axis=1)
df.head(10)


# feature importance
fi = clf_dt.feature_importances_
predictors = [x for x in df.iloc[:, 0:20].columns]
feat_imp = pd.Series(fi, predictors).sort_values(ascending=False)
feat_imp.plot(kind='bar', title='Feature Importances')
plt.ylabel('Feature Importance Score')

sub = pd.DataFrame({"Attribute": df.iloc[:, 0:20].columns, 
                    "Coefficient": clf_dt.feature_importances_,
                    "Impact Index": clf_dt.feature_importances_/(np.mean(clf_dt.feature_importances_))*100}).sort_values(by='Impact Index', ascending=False)
sub.to_csv('C:/Users/Jie.Hu/Desktop/Driver Analysis/0420/DA_outputs_dt.csv', index=False)



from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus

feature_cols = df.iloc[:,0:20].columns

dot_data = StringIO()

export_graphviz(clf_dt, 
                out_file=dot_data,  
                filled=True, 
                rounded=True,
                special_characters=True, 
                feature_names = feature_cols, 
                class_names=['Other','Top Box'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('explore.png')
Image(graph.create_png())








