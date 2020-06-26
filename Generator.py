# Assumption: Image is square
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import transforms as trs
import matplotlib as mpl


def InfoMap(k=200, n=20, r=10):
    loc = []
    for _ in range(n):
        loc.append([np.random.randint(0, k), np.random.randint(0, k),
                    np.random.randint(1, r)])
    return loc


if __name__ == '__main__':
    k, n, max_r = 200, 20, 10
    myDpi = 400
    mpl.rcParams['figure.dpi'] = myDpi
    RawMap = InfoMap()
    print(RawMap)
    fig = plt.figure(figsize=(6, 6), dpi=myDpi, facecolor='w')
    ax = fig.add_axes([0, 0, 1, 1])
    plt.autoscale(False, tight=True)
    plt.xlim(0, k)
    plt.ylim(0, k)

    for dot in RawMap:
        ax.add_artist(plt.Circle((dot[0], dot[1]), dot[2], color='r', alpha=0.5))
    ax.add_artist(plt.Circle((0, 0), 6, color='b', alpha=0.5))
    ax.add_artist(plt.Circle((k, k), 6, color='b', alpha=0.5))  # verifying graph is correctly extracted
    ax.axis('square', adjustabble='box-force')
    region = ax.transData.transform([(0, 0), (k, k)])
    ax.plot()
    BBOX = trs.Bbox(region / myDpi)
    plt.show()
    fig.savefig("testing1", dpi=myDpi, facecolor='w', bbox_inches=BBOX,
                pad_inches=0)
