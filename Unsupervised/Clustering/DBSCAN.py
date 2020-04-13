# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 19:04:44 2020

@author: Jie.Hu
"""


# dbscan clustering
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.cluster import DBSCAN
from collections import Counter

# define dataset
X, _ = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=4)
# define the model
model = DBSCAN(eps=0.30, min_samples=9)
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


#====

# Starting a tally of total iterations
dbscan_cluster  = []
count_cluster   = []

eps_space = np.arange(0.1, 5, 0.1)
min_samples_space = np.arange(1, 20, 1)
min_clust = 2
max_clust = 4

# Looping over each combination of hyperparameters
for eps_val in eps_space:
    for samples_val in min_samples_space:

        dbscan_grid = DBSCAN(eps = eps_val,
                             min_samples = samples_val)

        clusters = dbscan_grid.fit_predict(X)
        cluster_count = Counter(clusters)
        n_clusters = len(np.unique(clusters))

        if n_clusters >= min_clust and n_clusters <= max_clust:

            dbscan_cluster.append([eps_val,
                                   samples_val,
                                   n_clusters])

            count_cluster.append(cluster_count)


# last fit
model = DBSCAN(eps=0.1, min_samples=18)
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
plt.legend(clusters)
plt.show()
