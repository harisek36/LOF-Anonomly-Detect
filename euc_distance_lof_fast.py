"""
Dylan Slack

More efficient version of LOF using KD trees.
"""

import numpy as np
import time
from sklearn.neighbors import KDTree
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from dgen import data_styles

"""LOF function takes data set and K value and returns predicted outliers. 
   I used http://www.bistaumanga.com.np/blog/lof/ to help me figure out how to
   apply KD trees.""" 
def lof(X, k, outlier_threshold = 1.5, verbose = False):

    """Knn with KD trees"""
    start = time.time()
    BT = KDTree(X, leaf_size=k, p=2)

    distance, index = BT.query(X, k)
    distance, index = distance[:, 1:], index[:, 1:] 
    radius = distance[:, -1]

    """Calculate LRD."""
    LRD = np.mean(np.maximum(distance, radius[index]), axis=1)
    r = 1. / np.array(LRD)

    """Calculate outlier score."""
    outlier_score = np.sum(r[index], axis=1) / np.array(r, dtype=np.float16)
    outlier_score *= 1. / k

    # print ('Compute time: %g seconds.' % ((time.time() - start)))

    if verbose: print ("Recording all outliers with outlier score greater than %s."\
     % (outlier_threshold))

    outliers = []
    """ Could parallelize this for loop, but really not worth the overhead...
        Would get insignificant performance gain."""
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
    k = 15

    """Test data specificiations."""
    data_dim = 2
    num_tests = 1000
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

    if data_dim == 2:
        data_visualization(X, predicted_outliers) 

if __name__ == "__main__":
    main()






