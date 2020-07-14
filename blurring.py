import numpy as np
import cv2


def salt_noisy(image, density=0.04, portion=0.5):
    s_vs_p = portion
    amount = density
    out = image.copy()

    # Salt mode
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
    out[tuple(coords)] = 1

    # Pepper mode
    num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
    out[tuple(coords)] = 0
    return out


if __name__ == '__main__':
    img0 = cv2.imread('GT-testing1.png')
    img1 = salt_noisy(img0)
    cv2.imwrite('GT-testing1_blurred.png', img1)
    # save a copy of the ground truth
    binmap = cv2.inRange(img0, np.array([255, 255, 255]), np.array([255, 255, 255]))
    cv2.imwrite('GT-testing1_bw.png', binmap)
