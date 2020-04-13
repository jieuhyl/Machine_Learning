# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 23:16:25 2020

@author: Jie.Hu
"""


# gaussian mixture clustering
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture

# define dataset
X, _ = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=4)
# define the model
model = GaussianMixture(n_components=2)
# fit the model
model.fit(X)
# assign a cluster to each example
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


# choose n
n_components = np.arange(2, 6)
models = [GaussianMixture(n, covariance_type='full', random_state=1337).fit(X)
          for n in n_components]

plt.figure(figsize = (12, 8))
plt.plot(n_components, [m.aic(X) for m in models], label='BIC')
plt.plot(n_components, [m.bic(X) for m in models], label='AIC')
plt.legend(loc='best')
plt.xlabel('n_components')
plt.xticks(n_components)

final_model = GaussianMixture(n_components[[m.bic(X) for m in models].index(min([m.bic(X) for m in models]))], 
                                            covariance_type='full', random_state=1337).fit(X)

yhat = final_model.fit_predict(X)
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




# advanced
#================================================
ddgmm = BayesianGaussianMixture(n_components=2, 
                                covariance_type='full', 
                                weight_concentration_prior=100, 
                                weight_concentration_prior_type="dirichlet_distribution", 
                                max_iter=100, 
                                random_state=1337).fit(X)
yhat = ddgmm.fit_predict(X)
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


dpgmm = BayesianGaussianMixture(n_components=2, 
                                covariance_type='full',
                                weight_concentration_prior=100,
                                weight_concentration_prior_type='dirichlet_process', 
                                max_iter=100, 
                                random_state=1337).fit(X)
yhat = dpgmm.fit_predict(X)
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

