import Simulator as Sim
import numpy as np
from joblib import Parallel, delayed
from scipy.ndimage import gaussian_filter


# If the parameter for parallel processing is not correctly met,
# we will use only 1 processor
def check_parallel(ImgSize, numJobs, boxSize0):
    if ImgSize % (numJobs * boxSize0) == 0:
        return True
    return False


# cv2 uses GBR instead of RGB. and for any color, we let R be the "ROI" degree of certainty
# and GB be the degree of uncertainty. More certainty means more information of interest
# according to the NN. The information here, however, is not normalized and will be normalized later.

# The formula for calculating information is: (R-(B+G)/2)/255, and the information cannot be negative
def calculate_info(index, GT_Img0, n_worker0, boxSize0):
    base = index * int(GT_Img0.shape[0] / n_worker0)
    end = (index + 1) * int(GT_Img0.shape[0] / n_worker0)

    chunkRow = int(GT_Img0.shape[0] / boxSize0 / n_worker)
    chunkCol = int(GT_Img0.shape[1] / boxSize0)
    GT_InfoMap0 = np.zeros((chunkRow, chunkCol), dtype=float)

    for i in range(base, end):
        for j in range(GT_Img0.shape[1]):
            loc = GT_Img0[i, j]
            val = (loc[2] - (float(loc[0]) + loc[1]) / 2) / 255

            if val > 0:
                GT_InfoMap0[int((i - base) / boxSize0), int(j / boxSize0)] += val

    return GT_InfoMap0


if __name__ == '__main__':
    # Run after blurring.py
    # reduce the resolution and setting gaussian blurry level
    rows = 200
    cols = 200
    boxSize = 10
    gaussian_sigma = 1

    # Reduce resolution with ICRSsimulator
    GroundTruth = Sim.ICRSsimulator('testing1_blurred.png')
    if not GroundTruth.loadImage():
        print("Error: could not load image")
        exit(0)
    GroundTruth.setMapSize(rows, cols)
    GroundTruth.createMap()
    GT_Img = GroundTruth.img

    # Parallel implementation of calculate_info
    n_worker = 4
    results = None
    if check_parallel(GT_Img.shape[0], n_worker, boxSize):
        results = Parallel(n_jobs=n_worker)(
            delayed(calculate_info)(ind, GT_Img, n_worker, boxSize) for ind in range(n_worker))
    else:  # not tested
        results = Parallel(n_jobs=1)(
            delayed(calculate_info)(ind, GT_Img, n_worker, boxSize) for ind in range(n_worker))

    GT_InfoMap = results[0]
    for n in range(1, n_worker):
        GT_InfoMap = np.concatenate((GT_InfoMap, results[n]), axis=0)

    # Apply Gaussian blurry
    InfoMap2 = gaussian_filter(GT_InfoMap, sigma=gaussian_sigma)

    # Save the reduced resolution map with and without being blurred.
    np.save('InfoMap', np.asarray(GT_InfoMap), allow_pickle=False)
    np.save('InfoMap_blurred', np.asarray(InfoMap2), allow_pickle=False)
