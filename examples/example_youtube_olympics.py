# -*- coding: utf-8 -*-

import os
import sys
sys.path.append('../')
from pytube import YouTube
from annotator import Annotator

'''This example downloads a youtube video from the Olympic games, splits it into
several clips and let you annotate it'''

# Set up some folders
demo_folder = r'./'
clips_folder = r'./youtube_clips'
youtube_filename = 'youtube.mp4'

# Create the folders
if not os.path.exists(demo_folder):
    os.mkdir(demo_folder)
if not os.path.exists(clips_folder):
    os.mkdir(clips_folder)
    
# Download from youtube: "Women's Beam Final - London 2012 Olympics"
if not os.path.exists(os.path.join(demo_folder, 'youtube.mp4')):
    yt = YouTube('https://www.youtube.com/watch?v=VZvoufQy8qc')
    stream = yt.streams.filter(res='144p', mime_type='video/mp4').first()
    print('Downloading youtube file. This may take a while.\n' +
          'Let\'s be honest, this _will_ take a while...')
    stream.download(demo_folder, filename='youtube')
    
# Initialise the annotator
annotator = Annotator([
        {'name': 'result_table', 'color': (0, 1, 0)},
        {'name': 'olympics_logo', 'color': (0, 0, 1)},
        {'name': 'stretching', 'color': (0, 1, 1)}],
        clips_folder, N_show_approx=100, annotation_file='demo_labels.json')

# Split the video into clips
print('Generating clips from the video...')
annotator.video_to_clips(youtube_filename, clips_folder, clip_length=90, overlap=0, resize=0.5)

# Run the annotator
annotator.main()