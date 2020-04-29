# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 00:52:55 2020

@author: Jie.Hu
"""


# example of combining random oversampling and undersampling for imbalanced data
from collections import Counter
from sklearn.datasets import make_classification
# define dataset
X, y = make_classification(n_samples=10000, weights=[0.99], flip_y=0)
# summarize class distribution
print(Counter(y))



# random sampling =============================================================
''' over '''
from imblearn.over_sampling import RandomOverSampler
# define oversampling strategy
over = RandomOverSampler(sampling_strategy=1)
# fit and apply the transform
X, y = over.fit_resample(X, y)
# summarize class distribution
print(Counter(y))


''' under'''
from imblearn.under_sampling import RandomUnderSampler
# define undersampling strategy
under = RandomUnderSampler(sampling_strategy=1)
# fit and apply the transform
X, y = under.fit_resample(X, y)
# summarize class distribution
print(Counter(y))


''' over and under '''
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
# define oversampling strategy
over = RandomOverSampler(sampling_strategy={0:9900, 1:500})
# fit and apply the transform
X, y = over.fit_resample(X, y)
# summarize class distribution
print(Counter(y))
# define undersampling strategy
under = RandomUnderSampler(sampling_strategy={0:500, 1:500})
# fit and apply the transform
X, y = under.fit_resample(X, y)
# summarize class distribution
print(Counter(y))



# advanced undersampling ======================================================
''' ALLKNN '''
from imblearn.under_sampling import AllKNN, NeighbourhoodCleaningRule
# define undersampling strategy
under_allknn = AllKNN()
# fit and apply the transform
X, y = under_allknn.fit_resample(X, y)
# summarize class distribution
print(Counter(y))



''' RENN '''
from imblearn.under_sampling import CondensedNearestNeighbour, EditedNearestNeighbours, RepeatedEditedNearestNeighbours
# define undersampling strategy
under_renn = RepeatedEditedNearestNeighbours()
# fit and apply the transform
X, y = under_renn.fit_resample(X, y)
# summarize class distribution
print(Counter(y))



# advanced oversampling =======================================================
''' ADASYN '''
from imblearn.over_sampling import ADASYN
# define oversampling strategy
over_ada = ADASYN(random_state=42)
# fit and apply the transform
X, y = over_ada.fit_resample(X, y)
# summarize class distribution
print(Counter(y))


''' SMOTE '''
from imblearn.over_sampling import SMOTE, SMOTENC, BorderlineSMOTE
# define oversampling strategy
over_sm = SMOTE(random_state=42)
# fit and apply the transform
X, y = over_sm.fit_resample(X, y)
# summarize class distribution
print(Counter(y))



# advanced combine sampling ===================================================
''' SMOTEENN '''
from imblearn.combine import SMOTEENN
# define oversampling strategy
comb_sm = SMOTEENN(sampling_strategy=0.1)
# fit and apply the transform
X, y = comb_sm.fit_resample(X, y)
# summarize class distribution
print(Counter(y))



''' SMOTETomek '''
from imblearn.combine import SMOTETomek
# define oversampling strategy
comb_st = SMOTETomek(sampling_strategy={0:9900, 1:500})
# fit and apply the transform
X, y = comb_st.fit_resample(X, y)
# summarize class distribution
print(Counter(y))





