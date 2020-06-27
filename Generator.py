# Assumption: Image is square
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import transforms as trs
import matplotlib as mpl


def feature_gen(k=200, n=20, r=10):
    loc = []
    for _ in range(n):
        loc.append([np.random.randint(0, k), np.random.randint(0, k),
                    np.random.randint(1, r)])
    return loc


if __name__ == '__main__':
    k1, n1, max_r1 = 200, 20, 10
    myDpi = 400
    mpl.rcParams['figure.dpi'] = myDpi
    RawMap = feature_gen(k1, n1, max_r1)
    print(RawMap)
    fig = plt.figure(figsize=(6, 6), dpi=myDpi, facecolor='w')
    ax = fig.add_axes([0, 0, 1, 1])
    plt.autoscale(False, tight=True)
    plt.xlim(0, k1)
    plt.ylim(0, k1)

    for dot in RawMap:
        ax.add_artist(plt.Circle((dot[0], dot[1]), dot[2], color='r', alpha=0.5))
    # verifying graph is correctly extracted
    ax.axis('square', adjustabble='box-force')
    region = ax.transData.transform([(0, 0), (k1, k1)])
    ax.plot()
    BBOX = trs.Bbox(region / myDpi)

    fig.savefig("GC-testing1", dpi=myDpi, facecolor='w', bbox_inches=BBOX,
                pad_inches=0)
    Obstacle1 = feature_gen(k1, 8, 15)
    for dot in Obstacle1:
        ax.add_artist(plt.Circle((dot[0], dot[1]), dot[2], color='b', alpha=0.3))
    ax.plot()
    plt.show()
    fig.savefig("testing1", dpi=myDpi, facecolor='w', bbox_inches=BBOX,
                pad_inches=0)
