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
    myDpi = 200
    mpl.rcParams['figure.dpi'] = myDpi
    RawMap = InfoMap()
    print(RawMap)
    fig = plt.figure(figsize=(6, 6), dpi=200, facecolor='w')
    ax = fig.add_axes([0, 0, 1, 1])
    plt.autoscale(False, tight=True)
    plt.xlim(0, 200)
    plt.ylim(0, 200)
    plt.hlines(y=200, xmin=0, xmax=200, color="black")
    plt.hlines(y=0, xmin=0, xmax=200, color="black")
    plt.vlines(x=0, ymin=0, ymax=200, color="black")
    plt.vlines(x=200, ymin=0, ymax=200, color="black")
    for dot in RawMap:
        ax.add_artist(plt.Circle((dot[0], dot[1]), dot[2], color='r', alpha=0.5))
    ax.add_artist(plt.Circle((0, 0), 6, color='r', alpha=0.5))
    ax.axis('square', adjustabble='box-force')
    region = ax.transData.transform([(0, 0), (200, 200)])
    print(region)
    ax.plot()
    BBOX = trs.Bbox(region / myDpi)

    plt.show()
    fig.savefig("testing1", dpi=myDpi, facecolor='w', bbox_inches=BBOX,
                pad_inches=0)
