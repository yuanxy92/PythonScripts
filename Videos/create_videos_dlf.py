import cv2
import numpy as np
from glob import glob
import os.path

image_dir = '/Users/yuanxy/Downloads/data_0428_all/data_0428_7'
output_video_name = '0428_7'

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
(h, w) = cv2.imread(glob(f'{image_dir}/*.jpg')[0]).shape[:2]

out_writer = []
for i in range(4):
    out_writer.append(cv2.VideoWriter(f'{output_video_name}_camera_{i}.mp4', fourcc, 60, (w, h), 1))

for i in range(700):
    for j in range(4):
        imgname = f'{image_dir}/camera_{j*2}_frame_{i+1}.jpg'
        if os.path.isfile(imgname):
            img = cv2.imread(imgname)
            out_writer[j].write(img)

for i in range(4):
    out_writer[i].release()