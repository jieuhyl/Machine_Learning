# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 01:23:55 2020

@author: Jie.Hu
"""



''' model '''
# get mtx
X = df.iloc[:, 2:].values
y = df.iloc[:,1].values



''' 7: AdaBoost '''
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, classification_report
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
clf_ada = AdaBoostClassifier(random_state = 1337)
clf_ada.fit(X, y)


cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1337)
acc = cross_val_score(estimator = clf_ada, X = X, y = y, cv = cv, scoring='f1')
acc.mean(), acc.std()


parameters = {'algorithm':['SAMME', 'SAMME.R'],
              'n_estimators':[50,100,200],
              'learning_rate':[0.1,0.5,1,2,5]}
                                       
grid_search = GridSearchCV(estimator = clf_ada,
                           param_grid = parameters,
                           scoring='f1',
                           cv = cv,
                           n_jobs = -1)
start_time = time.time()
grid_search = grid_search.fit(X, y)
print('Training time: {} minutes'.format(round((time.time() - start_time)/60, 2)))
grid_search.best_params_, grid_search.best_score_


# last step
clf_ada = AdaBoostClassifier(algorithm = 'SAMME.R',
                             n_estimators = 50, 
                             learning_rate = 1,
                             random_state= 1337 )
clf_ada.fit(X, y)
y_pred = clf_ada.predict(X)

acc = cross_val_score(estimator = clf_ada, X = X, y = y, cv = cv, scoring='f1')
acc.mean(), acc.std()

print(classification_report(y, y_pred))


# feature importance
fi = clf_ada.feature_importances_
predictors = [x for x in df.iloc[:, 2:].columns]
feat_imp = pd.Series(fi, predictors).sort_values(ascending=False)
feat_imp.plot(kind='bar', title='Feature Importances')
plt.ylabel('Feature Importance Score')

sub = pd.DataFrame({"Attribute": df.iloc[:, 2:].columns, 
                    "Coefficient": clf_ada.feature_importances_,
                    "Impact Index": clf_ada.feature_importances_/(np.mean(clf_ada.feature_importances_))*100}).sort_values(by='Impact Index', ascending=False)
sub.to_csv('C:/Users/Jie.Hu/Desktop/Driver Analysis/0420/DA_outputs_ada.csv', index=False)
