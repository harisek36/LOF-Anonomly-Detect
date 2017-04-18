import arff
import numpy as np
from lof_fast import lof

a_data = arff.load(open('Parkinson/Parkinson_withoutdupl_05_v01.arff', 'rb'))

print (len(a_data))
X = np.asarray(list(a_data))
print (len(X))
#is_outlier = X[:,]


#print (l_data[0][23])