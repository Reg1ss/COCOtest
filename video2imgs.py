# -*- coding: utf-8 -*-
import cv2
import glob

vc = cv2.VideoCapture("/home/kuangrx/TopVT/video/D1_Al Salam_st_W_31_C-20180424-090000.mp4")
c=0
rval=vc.isOpened()
#timeF = 1  #视频帧计数间隔频率
while rval:
    c = c + 1
    rval, frame = vc.read()
#    if(c%timeF == 0): #每隔timeF帧进行存储操作
#        cv2.imwrite('smallVideo/smallVideo'+str(c) + '.jpg', frame) #存储为图像
    if c<500:
        cv2.imwrite('/home/kuangrx/TopVT/video/videoImgs/' + str(c).zfill(8) + '.jpg', frame) #存储为图像
        cv2.waitKey(1)
    else:
        break
vc.release()
