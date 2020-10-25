import cv2
import numpy as np
import os

dir = 'D:\Drown_which-cannot-fly\Belief_InfoFull\png\\'
for file in os.listdir(dir):
    img = cv2.imread(dir + file, 0)
    multiplicity = 40

    dim = (int(2000 / multiplicity), int(2000 / multiplicity))
    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    '''
    cv2.imwrite(
        'D:\Drown_which-cannot-fly\Belief_InfoLow\\' + str(multiplicity) +
        'x\png\\' + file[0:12] + 'Low_' + file[17:], resized)'''

    max_val = np.max(resized)
    resized = np.array(resized, dtype='float16') / max_val
    np.save('D:\Drown_which-cannot-fly\Belief_InfoLow\\' + str(multiplicity) +
            'x\\npy\\' + file[0:12] + 'Low_' + file[17:-4], resized)
