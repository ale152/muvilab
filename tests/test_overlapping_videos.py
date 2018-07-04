# -*- coding: utf-8 -*-

# Create dummy video containing just numbers
import os
import shutil
import cv2
import numpy as np
import sys
sys.path.append('../')

filename = './dummy_digits.mp4'
video_size = (160, 120)

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
cap = cv2.VideoWriter(filename, fourcc, 10, video_size)

for i in range(1100):
    img = np.zeros((video_size[1], video_size[0], 3)).astype('uint8')
    cv2.putText(img, str(i), (10, video_size[1]//2), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (255,255,255), thickness=2)
    cap.write(img)
    
cap.release()

# Create the clips folder
clips_folder = 'test_overlap_clips'
if os.path.exists(clips_folder):
    shutil.rmtree(clips_folder)
os.makedirs(clips_folder)
    

# Test the annotator
from annotator import Annotator
# Initialise the annotator
annotator = Annotator([
        {'name': 'test_label_1', 'color': (0, 1, 0)},
        {'name': 'test_label_2', 'color': (0, 0, 1)},
        {'name': 'test_label_3', 'color': (0, 1, 1)}],
        clips_folder, loop_duration=2, annotation_file='overlap_annotation.json',
        status_file='overlap_status.json')
# Create the overlapping clips
annotator.video_to_clips('dummy_digits.mp4', clips_folder, resize=0.5, overlap=0.5, clip_length=6)
# Run!
annotator.main()
