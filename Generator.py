# Assumption: Image is square
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import transforms as trs
from matplotlib.patches import Ellipse
import matplotlib as mpl


def enhance():
    return ['#ff007f', 0.2]


def real():
    return ['r', 0.5]


def reduce():
    return ['b', 0.2]


def feature_gen(k=200, n=20, r=15):
    loc = []
    for _ in range(n):
        loc.append([np.random.randint(0, k), np.random.randint(0, k),
                    np.random.randint(1, r), np.random.randint(0, 180)])
    return loc


if __name__ == '__main__':
    # Edit graphic parameters here
    k1, n1, max_r1 = 200, 10, 20
    myDpi = 400
    mpl.rcParams['figure.dpi'] = myDpi

    # 5.5 = 2000 pixel at dpi = 200. May scale up as demanded
    fig = plt.figure(figsize=(5.5, 5.5), dpi=myDpi, facecolor='w')
    ax = fig.add_axes([0, 0, 1, 1])
    plt.autoscale(False, tight=True)
    plt.xlim(0, k1)
    plt.ylim(0, k1)


    def load_feature(feature, shape='circle', target=None):

        if target is None:
            target = real()

        for dot in feature:
            if shape == 'circle':
                ax.add_artist(plt.Circle((dot[0], dot[1]), dot[2], ec='none', color=target[0], alpha=target[1]))
            elif shape == 'ellipse' or 'oval':

                # Edit parameters here:
                width_height_ratio = 3 / 4
                num_steps = dot[2] * 2

                for steps in range(1, num_steps + 1):
                    coefficient = steps / num_steps * 2
                    ax.add_artist(
                        Ellipse((dot[0], dot[1]), width=dot[2] * coefficient,
                                height=dot[2] * width_height_ratio * coefficient, angle=dot[3],
                                edgecolor=None, facecolor=target[0], alpha=target[1] * 2 / num_steps))


    RawMap = feature_gen(k1, n1, max_r1)
    load_feature(RawMap, shape='oval')

    # verifying graph is correctly extracted
    ax.axis('square', adjustabble='box-force')
    ax.plot()

    region = ax.transData.transform([(0, 0), (k1, k1)])
    BBOX = trs.Bbox(region / myDpi)
    fig.savefig("GT-testing1", dpi=myDpi, facecolor='w', bbox_inches=BBOX,
                pad_inches=0)

    Obstacle1 = feature_gen(k1, 15, 15)
    fakeTarget = feature_gen(k1, 10, max_r1)

    load_feature(Obstacle1, target=reduce())
    load_feature(fakeTarget, target=enhance())
    ax.plot()
    plt.show()
    fig.savefig("testing1", dpi=myDpi, facecolor='w', bbox_inches=BBOX,
                pad_inches=0)
