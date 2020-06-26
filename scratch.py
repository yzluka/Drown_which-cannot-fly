import matplotlib.pyplot as plt
import numpy as np

# set up a figure
fig = plt.figure()
ax = fig.add_subplot(111)
x = np.arange(0, 10, 0.005)
y = np.exp(-x / 2.) * np.sin(2 * np.pi * x)
ax.plot(x, y)
plt.show()
# what's one vertical unit & one horizontal unit in pixels?
print( ax.transData.transform((0, 0))
      )# Returns:
# array([[   0.,  384.],   <-- one y unit is 384 pixels (on my computer)
#        [ 496.,    0.]])  <-- one x unit is 496 pixels.
