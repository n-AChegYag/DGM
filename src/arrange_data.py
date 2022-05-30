import os
import cv2
from tqdm import tqdm

if __name__ == '__main__':

    RGB_PATH = '/home/ps/disk12t/ACY/AD_DGM/data/normal/3150FinalRGBBase'
    SAVE_PATH = '/home/ps/disk12t/ACY/AD_DGM/data/eye/N'
    
    all_images = os.listdir(RGB_PATH)
    for image in tqdm(all_images):
        rgb = cv2.imread(os.path.join(RGB_PATH, image))
        green_channel = rgb[:,:,1]
        cv2.imwrite(os.path.join(SAVE_PATH, image.split('.')[0] + '.bmp'), green_channel)
