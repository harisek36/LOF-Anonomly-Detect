import numpy as np
import matplotlib.pyplot as plt
linalg = np.linalg

N = 1000
mean = [1,1]
cov = [[0.3, 0.2],[0.2, 0.2]]
data = np.random.multivariate_normal(mean, cov, N)
L = linalg.cholesky(cov)
# print(L.shape)
# (2, 2)
uncorrelated = np.random.standard_normal((2,N))
data2 = np.dot(L,uncorrelated) + np.array(mean).reshape(2,1)
# print(data2.shape)
# (2, 1000) 
print (type(data))
#plt.scatter(data[:,0], data[:,1], c='yellow')
#plt.show()