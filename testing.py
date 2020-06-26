import Simulator as SIM
import numpy as np

sim = SIM.ICRSsimulator('D:\\Drown\\data\\18JUL15152008-S3DS_R07C1-011089227010_01_P001_scaled (1).TIF')

if not sim.loadImage():
    print("Error: could not load image")
    exit(0)

lower = np.array([100, 39, 100])
upper = np.array([200, 180, 150])

interestValue = 1  # Mark these areas as being of highest interest
sim.classify('Mining', lower, upper, interestValue)

lower = np.array([0, 20, 0])
upper = np.array([90, 157, 138])

interestValue = 0  # Mark these areas as being of no interest
sim.classify('Forest', lower, upper, interestValue)

lower = np.array([70, 39, 142])
upper = np.array([255, 255, 255])
interestValue = 0  # Mark these areas as being of no interest
sim.classify('Water', lower, upper, interestValue)

rows = 100
cols = 100
sim.setMapSize(rows, cols)

sim.createMap()
sim.showMap()
