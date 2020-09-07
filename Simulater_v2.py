import numpy as np
import matplotlib.pyplot as plt

edge_len = 2000


def get_oval_index(center, direction_angle, half_height, half_width):
    center = np.array(center)
    index = []
    buffer = max(half_width, half_height)
    w_start, w_end = center[0] - buffer, center[0] + buffer + 1
    h_start, h_end = center[1] - buffer, center[1] + buffer + 1
    half_width, half_height = int(half_width), int(half_height)

    for i in range(w_start, w_end):
        for j in range(h_start, h_end):
            # Oval formula with rotation angle integration
            distance = ((i - center[0]) * np.cos(direction_angle) + (j - center[1]) * np.sin(
                direction_angle)) ** 2 / half_width ** 2 + (
                               (i - center[0]) * np.sin(direction_angle) -
                               (j - center[1]) * np.cos(direction_angle)) ** 2 / half_height ** 2

            if distance <= 1:
                index.append(np.array([i, j]))

    index2 = []

    for each in index:

        if 0 <= each[0] < edge_len and 0 <= each[1] < edge_len:
            index2.append(each)

    index2 = np.array(index2, dtype='int16')

    return index2


def plot_oval(canvas_temp, center, direction_angle, half_height, half_width,
              peak, margin, oval_id, num_level, manual=False):
    # layer 0 is if
    # layer 1 is information
    if not 0 <= margin < peak <= 1:
        print("Please enter parameter margin and peak  properly")
    else:

        gradient_stride = np.floor(min(half_width, half_height) / num_level)
        delta = (peak - margin) / (np.floor(min(half_width, half_height) / gradient_stride))
        index = get_oval_index(center, direction_angle, half_height, half_width)

        if len(index[canvas_temp[index[:, 0], index[:, 1], 0] != 0]) == 0:
            canvas_temp[index[:, 0], index[:, 1], 0] = oval_id

        while min(half_width, half_height) > gradient_stride:
            ratio = half_height / half_width
            half_width -= gradient_stride
            half_height -= int(ratio * gradient_stride)
            index = get_oval_index(center, direction_angle, half_height, half_width)
            canvas_temp[index[:, 0], index[:, 1], 1] += delta
        else:
            if manual:
                print("Overlapping occurred")

    return canvas_temp


if __name__ == "__main__":
    # layer 0  is coloring, layer 1 is ID, and layer 2 is information
    canvas = np.zeros((edge_len, edge_len, 2), dtype='float32')
    canvas = plot_oval(canvas, [700, 800], np.pi / 4, 100, 60, 1, 0.4, 1, 8)
    canvas = plot_oval(canvas, [1400, 350], np.pi / 8, 120, 100, 1, 0.6, 2, 10)
    plt.imshow(canvas[:, :, 1] * 255, cmap='gray')
    plt.imshow(canvas[:, :, 0])
    plt.show()
