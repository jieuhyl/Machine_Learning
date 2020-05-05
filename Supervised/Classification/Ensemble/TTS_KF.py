# -*- coding: utf-8 -*-
"""
Created on Sun May  3 04:25:07 2020

@author: Jie.Hu
"""



#==============================================================================
# 1 tts
auc_preds = []
test_pred_prob = np.zeros(df_test.shape[0])
TTS = 0
for i in range(0, 5):
    train_x, test_x, train_y, test_y = train_test_split(X_train, y_train, test_size=0.2, random_state=i)
    
    model = clf_hgb_rand
    model.fit(train_x, train_y)
    oof_preds = model.predict_proba(test_x)[:, 1]
    test_pred_prob += model.predict_proba(df_test[predictors])[:, 1]/5
    print('TTS %2d AUC : %.6f' % (TTS + 1, roc_auc_score(test_y, oof_preds)))
    auc_preds.append(roc_auc_score(test_y, oof_preds))
    TTS += 1
print('Full AUC score %.6f' % np.mean(auc_preds))   

test_pred = (test_pred_prob >= 0.5).astype(bool)  
    
print('TTS_HGB ACC: %.3f' % accuracy_score(y_test, test_pred))
print('TTS_HGB F1: %.3f' % f1_score(y_test, test_pred))
print('TTS_HGB GMEAN: %.3f' % geometric_mean_score(y_test, test_pred))    
print('TTS_HGB AUC_ROC: %.3f' % roc_auc_score(y_test, test_pred_prob))


# 2 skf
skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 1337)
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
    n_fold = n_fold + 1
print('Full AUC score %.6f' % np.mean(auc_preds))   

test_pred = (test_pred_prob >= 0.5).astype(bool)  
    
print('SKF_HGB ACC: %.3f' % accuracy_score(y_test, test_pred))
print('SKF_HGB F1: %.3f' % f1_score(y_test, test_pred))
print('SKF_HGB GMEAN: %.3f' % geometric_mean_score(y_test, test_pred))    
print('SKF_HGB AUC_ROC: %.3f' % roc_auc_score(y_test, test_pred_prob))


# 3 rskf
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=1337)
auc_preds = []
test_pred_prob = np.zeros(df_test.shape[0])
n_fold = 0
for idx_train, idx_valid in rskf.split(X_train, y_train):
    train_x, train_y = X_train[idx_train], y_train[idx_train]
    valid_x, valid_y = X_train[idx_valid], y_train[idx_valid]
    
    model = clf_hgb_rand
    model.fit(train_x, train_y)
    oof_preds = model.predict_proba(valid_x)[:, 1]
    test_pred_prob += model.predict_proba(df_test[predictors])[:, 1]/25
    
    print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds)))
    auc_preds.append(roc_auc_score(valid_y, oof_preds))
    #del model, train_x, train_y, valid_x, valid_y
    #gc.collect()
    n_fold = n_fold + 1
print('Full AUC score %.6f' % np.mean(auc_preds))   

test_pred = (test_pred_prob >= 0.5).astype(bool)  
    
print('RSKF_HGB ACC: %.3f' % accuracy_score(y_test, test_pred))
print('RSKF_HGB F1: %.3f' % f1_score(y_test, test_pred))
print('RSKF_HGB GMEAN: %.3f' % geometric_mean_score(y_test, test_pred))    
print('RSKF_HGB AUC_ROC: %.3f' % roc_auc_score(y_test, test_pred_prob))
