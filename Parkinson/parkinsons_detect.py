import arff
import numpy as np
from lof_fast import lof

a_data = arff.load(open('Parkinson/Parkinson_withoutdupl_05_v01.arff', 'rb'))

"""Hope to compute anomolies across parkinson data set."""