import os
import cv2
import numpy as np
from tqdm import tqdm


if __name__ == '__main__':

    a_PATH = '/home/ps/disk12t/ACY/AD/MIA/data/pred/a'
    y_p_PATH = '/home/ps/disk12t/ACY/AD/MIA/data/pred/y_p'
    z_p_PATH = '/home/ps/disk12t/ACY/AD/MIA/data/pred/z_p'
    z_pp_PATH = '/home/ps/disk12t/ACY/AD/MIA/data/pred/z_pp'
    IMAGE_PATH = '/home/ps/disk12t/ACY/AD/MIA/data/all_a'
    A_PATH = '/home/ps/disk12t/ACY/AD/MIA/data/lh/a'
    Y_P_PATH = '/home/ps/disk12t/ACY/AD/MIA/data/lh/y_p'
    Z_P_PATH = '/home/ps/disk12t/ACY/AD/MIA/data/lh/z_p'
    Z_PP_PATH = '/home/ps/disk12t/ACY/AD/MIA/data/lh/z_pp'


    for i in tqdm(os.listdir(IMAGE_PATH)):
        img = cv2.imread(os.path.join(IMAGE_PATH, i))
        a = cv2.imread(os.path.join(a_PATH, i))
        y_p = cv2.imread(os.path.join(y_p_PATH, i))
        z_p = cv2.imread(os.path.join(z_p_PATH, i))
        z_pp = cv2.imread(os.path.join(z_pp_PATH, i))
        x, y = img.shape[0:2]
        A = cv2.resize(a, (y,x), interpolation=cv2.INTER_LINEAR)
        Y_P = cv2.resize(y_p, (y,x), interpolation=cv2.INTER_LINEAR)
        Z_P = cv2.resize(z_p, (y,x), interpolation=cv2.INTER_LINEAR)
        Z_PP = cv2.resize(z_pp, (y,x), interpolation=cv2.INTER_LINEAR)

        cv2.imwrite(os.path.join(A_PATH, i), A)
        cv2.imwrite(os.path.join(Y_P_PATH, i), Y_P)
        cv2.imwrite(os.path.join(Z_P_PATH, i), Z_P)
        cv2.imwrite(os.path.join(Z_PP_PATH, i), Z_PP)

