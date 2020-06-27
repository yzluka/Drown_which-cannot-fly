import Simulator as SIM
import numpy as np

sim = SIM.ICRSsimulator('D:\\Drown_which-cannot-fly\\testing1.png')

if not sim.loadImage():
    print("Error: could not load image")
    exit(0)

# lower = np.array([, 39, 100])
# upper = np.array([200, 180, 150])

# interestValue = 1  # Mark these areas as being of highest interest
# sim.classify('Mining', lower, upper, interestValue)

lower = np.array([255, 255, 255])
upper = np.array([255, 255, 255])

interestValue = 0  # Mark these areas as being of no interest
sim.classify('Background', lower, upper, interestValue)

lower = np.array([0, 0, 255])
upper = np.array([200, 200, 255])
interestValue = 0  # Mark these areas as being of no interest
sim.classify('target', lower, upper, interestValue)

rows = 100
cols = 100
sim.setMapSize(rows, cols)

sim.createMap()
sim.showMap()
