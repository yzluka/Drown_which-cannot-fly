# Produce NN-like output for information map
import numpy as np
import matplotlib.pyplot as plt
import cv2

box_pixel = 10 ** 2

# choose the output to be normalized
filename = 'GT-InfoMap'
myMap0 = np.load(filename + '.npy')
cv2.imwrite('GT-belief_map0.png', np.array(myMap0, dtype=np.uint8))
print('# of true non-empty block: ', len(np.argwhere(myMap0 > 0)))

filename = 'InfoMap_blurred'
myMap1 = np.load(filename + '.npy')
cv2.imwrite('belief_map0_blurred.png', np.array(myMap1, dtype=np.uint8))

# The two lines below defines the transformation algorithm
map2 = (myMap1 - box_pixel / 2) / 11
Sig_y = np.exp(map2) / (np.exp(map2) + 0.035) + 0.1
print('#number of significant block: ', len(np.argwhere(Sig_y > 0.6)))
# np.save(filename + '_sig_norm', Sig_y)
plt.hist(Sig_y, bins=10)
plt.show()
