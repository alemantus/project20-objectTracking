#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 12:53:13 2021

@author: alexander
"""

import os
import shutil


dest_file = "/python_samples/test_renamed_file.txt"

shutil.move

import os
i = 0
j = 0
k = 0
for filename in os.listdir("/home/alexander/Documents/DTU/Deep learning/project_20_data/project_20_data/video1/frames"):
    shutil.move(filename, "project_20_data/project_20_data/videoBoth/frames"+filename)
    i = i+1
    
for filename in os.listdir("/home/alexander/Documents/DTU/Deep learning/project_20_data/project_20_data/frames_video1"):
    shutil.move(filename, "project_20_data/project_20_data/videoBoth/frames_video/"+filename)
    
for filename in os.listdir("/home/alexander/Documents/DTU/Deep learning/project_20_data/project_20_data/video1/frames"):
    j = i + 1
    shutil.move(filename, "project_20_data/project_20_data/videoBoth/frames/"+"frame_"+str(j).zfill(6)+".xml")
    
for filename in os.listdir("/home/alexander/Documents/DTU/Deep learning/project_20_data/project_20_data/frames_video1"):
    k = i +1
    shutil.move(filename, "project_20_data/project_20_data/videoBoth/frames_video/"+"frame_"+str(k).zfill(6)+".PNG")
    
    