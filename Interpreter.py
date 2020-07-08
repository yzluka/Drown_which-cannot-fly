import Simulator as Sim
import numpy as np
from joblib import Parallel, delayed


def check_parallel(ImgSize, numJobs, boxSize0):
    if ImgSize % (numJobs * boxSize0) == 0:
        return True
    return False


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
    rows = 100
    cols = 100
    boxSize = 20
    GroundTruth = Sim.ICRSsimulator('GT-testing1.png')

    if not GroundTruth.loadImage():
        print("Error: could not load image")
        exit(0)
    '''
    DataImage = Sim.ICRSsimulator('testing1.png')
    if not DataImage.loadImage():
        print("Error: could not load image")
        exit(0)
    '''
    lower = np.array([255, 255, 255])
    upper = np.array([255, 255, 255])

    interestValue = 0  # Mark these areas as being of no interest
    GroundTruth.classify('Background', lower, upper, interestValue)

    lower = np.array([0, 0, 255])
    upper = np.array([200, 200, 255])
    interestValue = 0  # Mark these areas as being of no interest
    GroundTruth.classify('target', lower, upper, interestValue)

    GroundTruth.setMapSize(rows, cols)

    GroundTruth.createMap()
    GroundTruth.showMap()
    GT_Img = GroundTruth.img

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

    np.save('InfoMap', np.asarray(GT_InfoMap), allow_pickle=False)
