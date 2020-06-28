import Simulator as Sim
import numpy as np

GroundTruth = Sim.ICRSsimulator('GT-testing1.png')
if not GroundTruth.loadImage():
    print("Error: could not load image")
    exit(0)
Image = Sim.ICRSsimulator('testing1.png')
if not Image.loadImage():
    print("Error: could not load image")
    exit(0)

lower = np.array([255, 255, 255])
upper = np.array([255, 255, 255])

interestValue = 0  # Mark these areas as being of no interest
GroundTruth.classify('Background', lower, upper, interestValue)

lower = np.array([0, 0, 255])
upper = np.array([200, 200, 255])
interestValue = 0  # Mark these areas as being of no interest
GroundTruth.classify('target', lower, upper, interestValue)

rows = 100
cols = 100
GroundTruth.setMapSize(rows, cols)

GroundTruth.createMap()
GroundTruth.showMap()
