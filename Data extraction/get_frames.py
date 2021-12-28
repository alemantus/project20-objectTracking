# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 08:06:16 2021

@author: Alexander
"""
from os.path import exists
import cv2
import os
import shutil

video1_path = '../project_20_data/project_20_data/video1.avi'
video2_path = '../project_20_data/project_20_data/video2.avi'
object_categories = ['cola','beer']


cap= cv2.VideoCapture(video2_path)
i=0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    if(exists("../project_20_data/project_20_data/video2/frames/frame_"+str(i).zfill(6)+".xml")):
        #shutil.move('test.txt',dest_dir)
        #frame = cv2.resize(frame, (385,426))
        cv2.imwrite('/home/alexander/Documents/DTU/Deep learning/project_20_data/project_20_data/videoBoth/frames_video/frame_'+str(i+2717).zfill(6)+'.PNG',frame)
    i+=1
    print(i)

 
cap.release()
cv2.destroyAllWindows()