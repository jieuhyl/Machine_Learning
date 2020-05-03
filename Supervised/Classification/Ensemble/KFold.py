# -*- coding: utf-8 -*-
"""
Created on Sun May  3 04:25:07 2020

@author: Jie.Hu
"""



#==============================================================================
# 1 skf
skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 1337)
# Create arrays and dataframes to store results
auc_preds = []
test_pred_prob = np.zeros(df_test.shape[0])
n_fold = 0
for idx_train, idx_valid in skf.split(X_train, y_train):
    train_x, train_y = X_train[idx_train], y_train[idx_train]
    valid_x, valid_y = X_train[idx_valid], y_train[idx_valid]
    
    model = clf_hgb_rand
    
    model.fit(train_x, train_y)
    
    oof_preds = model.predict_proba(valid_x)[:, 1]
    test_pred_prob += model.predict_proba(df_test[predictors])[:, 1]/5 
    
    print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds)))
    auc_preds.append(roc_auc_score(valid_y, oof_preds))
    #del model, train_x, train_y, valid_x, valid_y
    #gc.collect()
    n_fold = n_fold + 1
print('Full AUC score %.6f' % np.mean(auc_preds))   

test_pred = (test_pred_prob >= 0.5).astype(bool)  
    
print('EN_HGB ACC: %.3f' % accuracy_score(y_test, test_pred))
print('EN_HGB F1: %.3f' % f1_score(y_test, test_pred))
print('EN_HGB GMEAN: %.3f' % geometric_mean_score(y_test, test_pred))    
print('EN_HGB AUC_ROC: %.3f' % roc_auc_score(y_test, test_pred_prob))



# 2 rskf
#kf = KFold(n_splits = 5, random_state = 1337, shuffle = True)
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=1337)
# Create arrays and dataframes to store results
auc_preds = []
test_pred_prob = np.zeros(df_test.shape[0])
n_fold = 0
for idx_train, idx_valid in rskf.split(X_train, y_train):
    train_x, train_y = X_train[idx_train], y_train[idx_train]
    valid_x, valid_y = X_train[idx_valid], y_train[idx_valid]
    
    model = clf_hgb_rand
    
    model.fit(train_x, train_y)
    
    oof_preds = model.predict_proba(valid_x)[:, 1]
    test_pred_prob += model.predict_proba(df_test[predictors])[:, 1]/10 
    
    print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds)))
    auc_preds.append(roc_auc_score(valid_y, oof_preds))
    #del model, train_x, train_y, valid_x, valid_y
    #gc.collect()
    n_fold = n_fold + 1
print('Full AUC score %.6f' % np.mean(auc_preds))   

test_pred = (test_pred_prob >= 0.5).astype(bool)  
    
print('EN_HGB ACC: %.3f' % accuracy_score(y_test, test_pred))
print('EN_HGB F1: %.3f' % f1_score(y_test, test_pred))
print('EN_HGB GMEAN: %.3f' % geometric_mean_score(y_test, test_pred))    
print('EN_HGB AUC_ROC: %.3f' % roc_auc_score(y_test, test_pred_prob))
