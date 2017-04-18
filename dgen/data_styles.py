"""
Copyright Dylan Slack
April 2017
"""

import numpy as np

"""Generate random clusters."""
def random_clusters(num_tests, num_outliers, dim, outliers = False):
	""" Generate cluster like data set."""
	X = 0.3 * np.random.randn(num_tests//2, dim)
	X_outliers = np.random.uniform(\
		low=-4, high=4, size=(num_outliers, dim))
	X = np.r_[X + 2, X - 2, X_outliers]

	if outliers:
		return X, X_outliers
	else: 
		return X

"""Generate multivariate normal data."""
def multivariate_normal(num_tests, mean, cov):
	X = np.random.multivariate_normal(mean, cov, num_tests)
	return X