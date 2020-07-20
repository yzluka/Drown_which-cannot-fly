# Produce NN-like output for information map
import numpy as np
import matplotlib.pyplot as plt

box_pixel = 10 ** 2

# choose the output to be normalized
filename = 'InfoMap'
myMap = np.load(filename + '.npy')

# The two lines below defines the transformation algorithm
map2 = (myMap - box_pixel / 2) / 5
Sig_y = np.exp(map2) / (np.exp(map2) + 1)

np.save(filename + '_sig_norm', Sig_y)
plt.hist(Sig_y, bins=10)
plt.show()
