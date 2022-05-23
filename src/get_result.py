import os
import cv2
import numpy as np
from tqdm import tqdm


if __name__ == '__main__':

    PRED_PATH = '/home/ps/disk12t/ACY/AD/MIA/data/pred/y_p'
    IMAGE_PATH = '/home/ps/disk12t/ACY/AD/MIA/data/all_a'
    REC_PATH = '/home/ps/disk12t/ACY/AD/MIA/data/result/rec'
    DIFF_PATH = '/home/ps/disk12t/ACY/AD/MIA/data/result/dif'

    for i in tqdm(os.listdir(PRED_PATH)):
        img = cv2.imread(os.path.join(IMAGE_PATH, i))
        pred = cv2.imread(os.path.join(PRED_PATH, i))
        x, y = img.shape[0:2]
        pred = cv2.resize(pred, (y,x), interpolation=cv2.INTER_LINEAR)
        diff = np.abs(pred.astype(np.int) - img.astype(np.int))
        cv2.imwrite(os.path.join(REC_PATH, i), pred)
        cv2.imwrite(os.path.join(DIFF_PATH, i), diff)

