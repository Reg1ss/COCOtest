# -*- coding: utf-8 -*-
import cv2
import glob
import os
fps = 25  # 保存视频的FPS，可以适当调整

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
videoWriter = cv2.VideoWriter('/home/kuangrx/TopTest/video/CD_Foggy.avi', fourcc, fps, (1920, 1080))  # 最后一个是保存图片的尺寸
imgs = glob.glob('/home/kuangrx/TopTest/chengpeng-dawu/*.jpg')
src_path  = '/home/kuangrx/TopTest/chengpeng-dawu'
img_list = os.listdir(src_path)
img_list.sort()
for img_index, imgname in enumerate(img_list):
    frame = cv2.imread(os.path.join(src_path, imgname))
    frame = cv2.resize(frame, (1920, 1080))
    videoWriter.write(frame)
    print ('{}/{}'.format(img_index, len(imgs)))
videoWriter.release()
