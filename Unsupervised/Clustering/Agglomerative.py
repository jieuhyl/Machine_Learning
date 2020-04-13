# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 18:20:09 2020

@author: Jie.Hu
"""


# agglomerative clustering
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch

# define dataset
X, _ = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=4)
# define the model

# affinity “euclidean”, “l1”, “l2”, “manhattan”, “cosine”,
# linkage{“ward”, “complete”, “average”, “single”
model = AgglomerativeClustering(n_clusters=2, affinity = 'cosine', linkage='complete')
# fit model and predict clusters
yhat = model.fit_predict(X)
# retrieve unique clusters
clusters = np.unique(yhat)
# create scatter plot for samples from each cluster
for cluster in clusters:
	# get row indexes for samples with this cluster
	row_ix = np.where(yhat == cluster)
	# create scatter of these samples
	plt.scatter(X[row_ix, 0], X[row_ix, 1])
# show the plot
plt.show()



# dendrogram
dendrogram = sch.dendrogram(sch.linkage(X, metric = 'euclidean', method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()