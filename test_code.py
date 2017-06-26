import numpy as np
col1 = np.genfromtxt('test.txt',usecols=(1),delimiter=' ',dtype=None)
col2 = np.genfromtxt('test.txt',usecols=(2),delimiter=' ',dtype=None)
print(col2)