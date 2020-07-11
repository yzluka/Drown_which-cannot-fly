from matplotlib.patches import Ellipse


def gradient_ellipse(ax, x, y, width, height, color, angle=0, steps=10):
    for w, h in zip(range(0, width, width / steps), range(0, height, height / steps)):
        ax.add_patch(Ellipse((x, y), width=w, height=h,
                             edgecolor='none', facecolor=color))
