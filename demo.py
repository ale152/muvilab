# -*- coding: utf-8 -*-
import os
import cv2
from pytube import YouTube
from annotator import Annotator
# The example is a youtube video. The file is downloaded, split into several clips
# and fed into the annotator.
demo_folder = r'C:\Users\am14795\Local Documents\demo'
clips_folder = r'C:\Users\am14795\Local Documents\demo\clips'
youtube_filename = 'youtube.mp4'
clip_filename = 'clip_%04d.mp4'

# Create the folders
if not os.path.exists(demo_folder):
        os.mkdir(demo_folder)
if not os.path.exists(clips_folder):
    os.mkdir(clips_folder)
    
# Download from youtube: "Women's Beam Final - London 2012 Olympics"
if not os.path.exists(os.path.join(demo_folder, 'youtube.mp4')):
    yt = YouTube('https://www.youtube.com/watch?v=VZvoufQy8qc')
    stream = yt.streams.filter(res='144p', mime_type='video/mp4').first()
    print('Downloading youtube file. Please wait...')
    stream.download(demo_folder, filename='youtube')

# Split the video into several clips of 90 frames each
if not os.path.exists(os.path.join(clips_folder, clip_filename % 0)):
    resize = 0.5
    clip_length = 90  # Frames
    clip_counter = 0
    video_time = 0
    # Open the source video
    cap = cv2.VideoCapture(os.path.join(demo_folder, youtube_filename))
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    fps = cap.get(cv2.CAP_PROP_FPS)
    init = True
    while cap.isOpened():
        _, frame = cap.read()
        frame = cv2.resize(frame, (0, 0), fx=resize, fy=resize)
        # Initialise the video
        if init:
            fdim = frame.shape
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            clip = cv2.VideoWriter(os.path.join(clips_folder, 'clip_%04d.mp4' % clip_counter),
                                   fourcc, fps, (fdim[1], fdim[0]))
            clip_time = 0
            video_time = 0
            init = False
            
        clip.write(frame)
        if clip_time < clip_length:
            clip_time += 1
        else:
            # Save the clip
            print('\rClip %d complete' % clip_counter, end=' ')
            clip.release()
            clip_time = 0
            clip_counter += 1
            clip = cv2.VideoWriter(os.path.join(clips_folder, 'clip_%04d.mp4' % clip_counter),
                                   fourcc, fps, (fdim[1], fdim[0]))
            
        if video_time < n_frames-1:
            video_time += 1
        else:
            cap.release()
            break
    
# Run the annotator
annotator = Annotator([
        {'name': 'result_table', 
        'color': (0, 1, 0),
        'event': cv2.EVENT_LBUTTONDOWN},

        {'name': 'olympics_logo', 
        'color': (0, 0, 1),
        'event': cv2.EVENT_LBUTTONDBLCLK},
         
         {'name': 'stretching', 
        'color': (0, 1, 1),
        'event': cv2.EVENT_MBUTTONDOWN}
        ], clips_folder, N_show_approx=100,
        annotation_file='demo_labels.json')

annotator.main()