"""
Numpy optimized version of LOF using euclidean distance to determine k nearest neighbors.

:license: GNU GPL v2
"""

import numpy as np
import time
from sklearn.neighbors import DistanceMetric
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from dgen import data_styles

"""LOF function takes data set and K value and returns predicted outliers. 
   I used http://www.bistaumanga.com.np/blog/lof/ to help me figure out how to 
   use a Euclidean distance metric with sklearn.""" 
def lof(X, k, outlier_threshold = 1.5, verbose = False):
    """Determine nearest neighbors using euclidean distance search."""
    dist = DistanceMetric.get_metric('euclidean').pairwise(X)
    sorted_d = np.argsort(dist, axis=1)
    k_nearest = sorted_d[:,1 : k + 1] 

    """Reachability distance for all points."""
    radius = np.linalg.norm(X - X[k_nearest[:, -1]], axis = 1) 

    """Local reachability distance computation."""
    LRD = []
    for i in range(k_nearest.shape[0]):
        LRD.append(np.mean(np.maximum(dist[i, k_nearest[i]], radius[k_nearest[i]])))
    r = 1. / np.array(LRD)

    """Compare reachability densities to generate outlier scores."""
    outlier_score = np.sum(r[k_nearest], axis = 1)/ np.array(r, dtype = np.float16)
    outlier_score *= 1. / k

    if verbose: print ("Recording all outliers with outlier score greater than %s."\
     % (outlier_threshold))

    """Find outliers greater than user given threshold."""
    outliers = []
    for i, score in enumerate(outlier_score):
        if score > outlier_threshold:
            outliers.append([X[i], score])

    if verbose:
        print ("Detected outliers:")
        print (outliers)

    return outliers

def data_visualization(X,X_outliers):
    """Plot data nicely."""
    plt.scatter(X[:,0], X[:,1], c='yellow')

    all_outliers = []
    scores = []
    for i, pair in enumerate(X_outliers):
        all_outliers.append(pair[0])
        scores.append(pair[1])

    X_o = np.vstack(all_outliers)
    
    plt.scatter(X_o[:,0], X_o[:,1], c='red')

    plt.show()


def main():
    """Set K nearest neighbors to look at."""
    k = 5

    """Test data specificiations."""
    data_dim = 2
    num_tests = 400
    num_outliers = 2

    mean = [1,1]
    cov = [[0.3, 0.2],[0.2, 0.2]]

    """Generate test data set.  Couple options here"""
    X = data_styles.random_clusters(num_tests,num_outliers,data_dim)
    #X = data_styles.multivariate_normal(num_tests,mean,cov)

    start = time.time()
    predicted_outliers = lof(X, k, outlier_threshold = 1.75)

    print ("---------------------")
    print ("Finding outliers in %s values took %s seconds." % (len(X),time.time() - start))
    print ("---------------------")

    #if data_dim == 2:
        #data_visualization(X, predicted_outliers) 

if __name__ == "__main__":
    main()






