import numpy as np
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

edge_len = 2000


def rotation(index, direction_angle0):
    index = np.array(index)
    index2 = []
    for i in index:
        sin = np.sin(direction_angle0)
        cos = np.cos(direction_angle0)
        index2.append(np.dot(np.array([[cos, -sin],
                                       [sin, cos]]), i))
    index2 = np.array(index2, dtype='int16')
    return index2


def add_oval(canvas_temp, center, direction_angle, half_height, half_width, center_info, edge_info, id):
    center = np.array(center)
    # First, check if in the oval
    index = []
    index2 = []
    half_width, half_height = int(half_width), int(half_height)
    for i in range(-half_width, half_width + 1):
        for j in range(-half_height, half_height + 1):
            if i ** 2 / half_width ** 2 + j ** 2 / half_height ** 2 <= 1:
                index.append(np.array([i, j]))

    index = rotation(np.array(index), direction_angle)
    for each in index:
        temp = each + center
        if 0 <= temp[0] < edge_len and 0 <= temp[1] < edge_len:
            index2.append(each)
    index2 = np.array(index2)
    print(index2)
    plt.plot(index2[:, 0], index2[:, 1], ls='', marker='o')
    plt.axis('equal')
    plt.show()

    print("index shape is: ", index.shape)

    '''
    # Second, transformation

    # Third, check overlapping
    for i in range(edge_len):
        temp[i] = np.sqrt((row_num - center[0]) ** 2 + (i - center[1]) ** 2)
    new_indexes = np.argwhere()
    '''


if __name__ == "__main__":
    # layer 0  is coloring, layer 1 is ID, and layer 2 is information
    canvas = np.zeros((edge_len, edge_len, 3), dtype='float32')
    add_oval(canvas, [100, 100], np.pi / 4, 15, 7, 0.9, 0.6, 1)
