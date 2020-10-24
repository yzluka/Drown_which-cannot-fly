import cv2
import numpy as np
from os import listdir

dir_name = 'D:\Drown_which-cannot-fly\\test_imgs\good_env_images\\'
file_name = 'GT-testing1'
for temp in listdir(dir_name):
    file_name = temp[0:-4]
    # read in grayscale
    rawImg = np.array(cv2.imread(dir_name + file_name + '.png', 0), dtype='float16')
    max_val = np.max(rawImg)
    print(max_val, rawImg.dtype)
    dir_name_out = 'D:\Drown_which-cannot-fly\GT_InfoFull\\'
    file_name_out = file_name + '_Info_Full'
    img_name_out = file_name_out + '.png'
    np.save(dir_name_out + file_name_out, rawImg)
    img_out = np.array(rawImg * 255, dtype='uint8')
    cv2.imwrite(dir_name_out + file_name_out + '.png', img_out)
