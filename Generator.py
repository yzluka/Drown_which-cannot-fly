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
    k1, n1, max_r1 = 200, 20, 20
    myDpi = 400
    mpl.rcParams['figure.dpi'] = myDpi

    # 5.5 = 2000 pixel at dpi = 200. May scale up as demanded
    fig = plt.figure(figsize=(5.5, 5.5), dpi=myDpi, facecolor='w')
    ax = fig.add_axes([0, 0, 1, 1])
    plt.autoscale(False, tight=True)
    plt.xlim(0, k1)
    plt.ylim(0, k1)


    def load_feature(feature, shape='circle', color='r', alpha=0.5):
        if shape == 'circle':
            for dot in feature:
                ax.add_artist(plt.Circle((dot[0], dot[1]), dot[2], ec='none', color=color, alpha=alpha))


    RawMap = feature_gen(k1, n1, max_r1)
    load_feature(RawMap)

    # verifying graph is correctly extracted
    ax.axis('square', adjustabble='box-force')
    ax.plot()

    region = ax.transData.transform([(0, 0), (k1, k1)])
    BBOX = trs.Bbox(region / myDpi)
    fig.savefig("GT-testing1", dpi=myDpi, facecolor='w', bbox_inches=BBOX,
                pad_inches=0)

    Obstacle1 = feature_gen(k1, 15, 15)
    fakeTarget = feature_gen(k1, 10, max_r1)

    load_feature(Obstacle1, color='b', alpha=0.3)
    load_feature(fakeTarget, color='#FF007F', alpha=0.5)
    ax.plot()
    plt.show()
    fig.savefig("testing1", dpi=myDpi, facecolor='w', bbox_inches=BBOX,
                pad_inches=0)
