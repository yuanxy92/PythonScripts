import cv2
import numpy as np
from glob import glob
import os.path

image_dir1 = '/Users/yuanxy/Seafile/goodview'
image_dir2 = '/Users/yuanxy/Seafile/randomview'
image_dir3 = '/Users/yuanxy/Seafile/origin'
output_video_name = 'random_view.avi'

fourcc = cv2.VideoWriter_fourcc('m', 'j', 'p', 'g')
(h, w) = cv2.imread(glob(f'{image_dir3}/*.png')[0]).shape[:2]
out = cv2.VideoWriter(output_video_name, fourcc, 5, (2*w, 2*h), 1)

for i in range(18):
    j = i
    while True:
        imgname1 = f'{image_dir1}/{j}.png'
        imgname2 = f'{image_dir2}/{j}.png'
        imgname3 = f'{image_dir3}/{j}.png'
        if os.path.isfile(imgname1) and os.path.isfile(imgname2) and os.path.isfile(imgname3):
            img1 = cv2.imread(imgname1)
            img2 = cv2.imread(imgname2)
            img3 = cv2.imread(imgname3)
            img = np.zeros((2*h,2*w,3), np.uint8)
            img[0:h,0:w] = img1   
            img[0:h,w:] = img2   
            img[h:,0:w] = img3   
            out.write(img)
        else:
            break
        j = j + 18

out.release()