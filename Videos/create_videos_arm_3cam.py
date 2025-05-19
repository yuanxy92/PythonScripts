import cv2
import numpy as np
from glob import glob
import os.path

image_dir = '/Users/yuanxy/Downloads/LocalSend'
output_video_name = 'video_20240824_3cam'

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
(h, w) = cv2.imread(glob(f'{image_dir}/arm_3cam_images/*.png')[0]).shape[:2]

out_writer = cv2.VideoWriter(f'{image_dir}/{output_video_name}.mp4', fourcc, 15, (w * 3, h), 1)

for i in range(2000):
    imgname1 = f'{image_dir}/arm_3cam_images/camera_1_frame_{i+1}_corrected.png'
    imgname2 = f'{image_dir}/arm_3cam_images/camera_2_frame_{i+1}_corrected.png'
    imgname3 = f'{image_dir}/arm_3cam_images/camera_3_frame_{i+1}_corrected.png'
    if os.path.isfile(imgname1) and os.path.isfile(imgname2) and os.path.isfile(imgname3):
        img1 = cv2.imread(imgname1)
        img2 = cv2.imread(imgname2)
        img3 = cv2.imread(imgname3)
        img = cv2.hconcat([img1, img2, img3])
        out_writer.write(img)

out_writer.release()