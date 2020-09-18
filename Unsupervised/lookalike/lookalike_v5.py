# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 02:15:47 2020

@author: Jie.Hu
"""



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.neighbors import NearestNeighbors
from datetime import datetime
import time


start_time = time.time()

# define dataset
X, y = make_classification(n_samples=2000, n_features=20, n_informative=15, weights=[0.9], n_redundant=5, n_classes=2, random_state=1337)
df = pd.DataFrame(data=X, columns=['V'+str(i) for i in range(1,20+1)])
df.insert(0, 'ID', np.array(range(2000)))
df['Target'] = y


# check data
#df.info()
#df.isnull().values.sum()


# mtx
X = df.iloc[:,1:21].values
y = df['Target'].values


# demension reduction 
nca = NeighborhoodComponentsAnalysis(random_state=1234)
X = nca.fit_transform(X, y)


# transformation
mms = MinMaxScaler()
X = mms.fit_transform(X)


def run_simulation(n_neighbors=10, metric='minkowski', thres1=10, thres2=None):
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric=metric).fit(X)

    # Let's find the k-neighbors of each point in object X. To do that we call the kneighbors() function on object X.
    distances, indices = nbrs.kneighbors(X)
        
    # get all the sub
    sub_id = df.loc[df.Target == 1, 'ID'].tolist()
    sub = indices[sub_id]
    
    # all condidates
    all_cand = {}
    for i in sub.tolist():
        for j in range(1, n_neighbors):
            all_cand[i[j]] = all_cand.get(i[j], 0) + 1
    
    # check sub distribution
    dist = []
    for i,j in all_cand.items():
        if i in sub_id:
            dist.append(j)
    #plt.hist(dist)
    #print(np.min(dist), np.mean(dist), np.median(dist), np.max(dist))
    
    
    # add thres1
    first_sub = []
    first_recommend = []
    for i,j in all_cand.items():
        if i in sub_id and j >= thres1:
            first_sub.append(i)
        if i not in sub_id and j >= thres1:
            first_recommend.append(i)

    
    if thres2 is None:
        ans_size_1 = len(first_sub)
        ans_percent_1 = round(len(first_sub)/len(sub_id), 4)
        res_size_1 = len(first_recommend)
        res_percent_1 = round(len(first_recommend)/(len(df)-len(sub_id)), 4)
        
        dct1 = dict()
        name = (n_neighbors, metric, thres1, thres2)
        dct1[name] = [ans_size_1, ans_percent_1, res_size_1, res_percent_1]
        
        return dct1, first_recommend
    
    else:
        connection = indices[first_recommend]
        # add thres2
        second_sub = set()
        second_recommend = set()
        for i in connection.tolist():
            for j in range(1, thres2):
                if i[j] in sub_id and i[j] not in first_sub:
                    second_sub.add(i[j])
                if i[j] not in first_recommend and i[j] not in sub_id:
                    second_recommend.add(i[j])
                    
        second_sub = list(second_sub)
        second_recommend = list(second_recommend)
        
        second_sub = second_sub + first_sub
        second_recommend = first_recommend + second_recommend
        
        ans_size_2 = len(second_sub)
        ans_percent_2 = round(len(second_sub)/len(sub_id), 4)
        res_size_2 = len(second_recommend)
        res_percent_2 = round(len(second_recommend)/(len(df)-len(sub_id)), 4)
        
        dct2 = dict()
        name = (n_neighbors, metric, thres1, thres2)
        dct2[name] = [ans_size_2, ans_percent_2, res_size_2, res_percent_2]
        
        return dct2, second_recommend
    
#run_simulation(50, 'minkowski', 5, 2)


# simulation
res = []
for a in [5, 10, 15, 20, 30, 40]:
    for b in ['minkowski', 'cosine']:
        for c in [3,7,11,15,20]:
            for d in [None, 2, 3]:
                ans, _ = run_simulation(a, b, c, d)
                res.append(ans)


# export
df_res = pd.DataFrame()
df_res['Setting'] = [list(r.keys())[0] for r in res]
df_res['Output'] = [list(r.values())[0] for r in res]

#df_res.set_index('setting')
df_res[['Cover_Size','Cover_Percentage','Recommend_Size','Recommend_Percentage']] = pd.DataFrame(df_res.Output.tolist(), index= df_res.index)
df_res.drop('Output', axis = 1, inplace = True)


# submission
now_time = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")

path = 'C:/Users/Jie.Hu/Desktop/'
file_name = 'Simulation_Result_' + now_time
export_path = path + file_name + '.csv'
df_res.to_csv(export_path, index=False)


# recommend
_, ans = run_simulation(40, 'minkowski', 3, 2)

# export
df_ans = pd.DataFrame({'Recommend_ID': ans})

file_name = 'Simulation_Recommend_' + now_time
export_path = path + file_name + '.csv'
df_ans.to_csv(export_path, index=False)

print('Running Time: {} minutes'.format(round((time.time() - start_time)/60, 2)))