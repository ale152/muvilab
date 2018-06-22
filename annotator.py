# -*- coding: utf-8 -*-

import numpy as np
import cv2
import os
import json
from shutil import copyfile

class Annotator():
    '''Annotate multiple videos simultaneously by clicking on them. The current configuration
    requires the videos to be in subfolders located in "videos_folder". The algorithm will loop
    through the folders and load all the videos in them.
    
    /!\ LIMITATIONS /!\ 
    This code was mainly written for my specific application and I decided to upload it on github
    as it might be helpful for other people.
    It assumes that all the videos are the same length (100 frames) and are black and white. I will
    try to update the code to make it more general, according to the time available and the number
    of requests.'''

    def __init__(self, labels):
        # Set the labels
        self.labels = labels

    
    def find_videos(self, videos_folder, video_ext):
        '''Loop over the video folder looking for video files'''
        videos_list = []
        for folder, _, files in os.walk(videos_folder):
            for file in files:
                fullfile_path = os.path.join(folder, file)
                if os.path.splitext(fullfile_path)[1] in video_ext:
                    videos_list.append(os.path.join(folder, file))
                    
        return videos_list
    
    def list_to_pages(self, videos_list):
        '''Split a list of videos into an array arranged by pages of mosaics'''
        N_pages = int(np.ceil(len(videos_list)/self.Nx/self.Ny))
        video_pages = [[[[] for _ in range(self.Nx)] for _ in range(self.Ny)] for _ in range(N_pages)]
        vid = 0
        for p in range(N_pages):
            for i in range(self.Ny):
                for j in range(self.Nx):
                    video_pages[p][i][j] = {'video': videos_list[vid],
                                            'label': ''}
                    if vid == len(videos_list)-1:
                        return video_pages
                    else:
                        vid += 1

    def create_mosaic(self, videos_list):
        '''This function create a mosaic of videos given a set of video files'''
        # List video files
        init = True
        # Loop over all the video files in the day folder
        for vi, video_file in enumerate(videos_list):
            print('\r', 'Loading file %s' % video_file, end=' ')
            
            # Deal with long lists
            if vi == self.Nx*self.Ny:
                print('The list of videos doesn\'t fit in the mosaic.')
                break
            
            # Open the video
            cap = cv2.VideoCapture(video_file)
            
            # Load the video frames
            while(cap.isOpened()):
                ret, frame = cap.read()
                
                # Initialise the video            
                if init:
                    fdim = frame.shape
                    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    mosaic = np.zeros((n_frames, fdim[0]*self.Ny, fdim[1]*self.Nx, 3))
                    i_scr = 0
                    j_scr = 0
                    k_time = 0
                    mosaic_names = [[[] for _ in range(self.Nx)] for _ in range(self.Ny)]
                    init = False
                
                # Add video to the grid
                mosaic[k_time, i_scr*fdim[0]:(i_scr+1)*fdim[0],
                             j_scr*fdim[1]:(j_scr+1)*fdim[1], :] = frame[... , :]/255
                             
                # Save the file name
                mosaic_names[i_scr][j_scr] = video_file
                
                # When all the frames have been read
                k_time += 1
                if k_time == n_frames:
                    cap.release()
                    k_time = 0
                    break
            
            # Increase the mosaic indices
            i_scr += 1
            if i_scr == self.Ny:
                i_scr = 0
                j_scr += 1
        
        # Write some metadata to yield with the batch
        return mosaic, mosaic_names

    
    def batch_generator(self, videos_folder, starting_day, starting_video, total_vid_ann):
        '''Generator that reads the videos in a folder and returns a mosaic of
        videos in the form of numpy array'''
        
        # List all days
        folder_list = os.listdir(videos_folder)
        folder_list.sort()
        # Remove empty folders
        folder_list = [bf for bf in folder_list if os.listdir(os.path.join(videos_folder, bf)) != []]
        # Start from the selected day
        if starting_day is not None:
            if starting_day in folder_list:
                first_index = folder_list.index(starting_day)
                folder_list = folder_list[first_index:]
        
        # Find total number of videos
        total_videos = np.sum([len(os.listdir(os.path.join(videos_folder, bf))) for bf in folder_list])
        
        # Loop over the days
        for day_folder in folder_list:
            print('Entering folder %s' % day_folder)
            # List video files
            videos_list = os.listdir(os.path.join(videos_folder, day_folder))
            videos_list.sort()
            init = True
            page = 0
            
            # Start from the selected day
            if starting_video is not None:
                if starting_video in videos_list:
                    first_index = videos_list.index(starting_video)
                    videos_list = videos_list[first_index:]
            
            # Loop over all the video files in the day folder
            for vi, video_file in enumerate(videos_list):
                print('\r', 'Loading file %s' % video_file, end=' ')
                
                # Open the video
                cap = cv2.VideoCapture(os.path.join(videos_folder, day_folder, video_file))
                total_vid_ann += 1
                
                # Load the video frames
                while(cap.isOpened()):
                    ret, frame = cap.read()
                    
                    # Initialise the video            
                    if init:
                        self.dim = frame.shape
                        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        mosaic = np.zeros((n_frames, self.dim[0]*self.Ny, self.dim[1]*self.Nx, 3))
                        i_scr = 0
                        j_scr = 0
                        k_time = 0
                        video_names = [[[] for _ in range(self.Nx)] for _ in range(self.Ny)]
                        init = False
                    
                    # Add video to the grid
                    mosaic[k_time, i_scr*self.dim[0]:(i_scr+1)*self.dim[0],
                                 j_scr*self.dim[1]:(j_scr+1)*self.dim[1], :] = frame[... , :]/255
                                 
                    # Save the file name
                    video_names[i_scr][j_scr] = video_file
                    
                    # When all the frames have been read
                    k_time += 1
                    if k_time == n_frames:
                        cap.release()
                        k_time = 0
                        break
                
                # Check if a label already exists
                existing_label = [bf for bf in self.annotations if bf['video'] == video_file]
                if existing_label != []:
                    color = [bf for bf in self.labels if bf['name'] == existing_label[0]['label']][0]['color'] # TODO This is horrible...
                    self.rectangle[i_scr][j_scr] = {'p1': (j_scr*self.dim[1], i_scr*self.dim[0], ),
                               'p2': ((j_scr+1)*self.dim[1], (i_scr+1)*self.dim[0]), 
                               'color': color, 'label': existing_label[0]['label']}
                
                # Increase the indices
                i_scr += 1
                if i_scr == self.Ny:
                    i_scr = 0
                    j_scr += 1
                
                # The page is full or there are no more videos in the folder
                if j_scr == self.Nx or vi == len(videos_list)-1:
                    # Write some metadata to yield with the batch
                    status = {'day': day_folder,
                            'last_video': video_names[0][0],
                            'count': total_vid_ann}
                    page += 1
                    print('\nShowing page %d of %d, %.2f%% completed' % (page, 
                                                     len(videos_list)//self.Nx//self.Ny + 1,
                                                     total_vid_ann/total_videos*100))
                    yield mosaic, video_names, status
                    
                    # Reset everything for the next page
                    init = True
                    if self.debug_verbose == 1:
                        print('Loading the next batch...')
                    
                    
    # Create the click callback
    def click_callback(self, event, x_click, y_click, flags, param):
        '''Click callback that sets the lables based on the click'''
        
        # Loop over the labels
        for label in self.labels:
            # Check the event
            if event == label['event']:
                self.set_label(label['name'], label['color'], x_click, y_click)

        # Detect right click to remove label
        if event == cv2.EVENT_RBUTTONDOWN:
            self.remove_label(x_click, y_click)


    def set_label(self, label_text, label_color, x_click, y_click):
        '''Set a specific label based on the user click input'''
        # Find the indices of the clicked sequence
        i_click = int(np.floor((y_click ) / self.mosaic_dim[1] * self.Ny))
        j_click = int(np.floor((x_click ) / self.mosaic_dim[2] * self.Nx))
        
        # Create the label
        self.video_pages[self.current_page][i_click][j_click]['label'] = label_text
        
        # Update the rectangles
        self.update_rectangles()


    def remove_label(self, x_click, y_click):
        '''Remove label from the annotations'''
        # Find the indices of the clicked sequence
        i_click = int(np.floor((y_click ) / self.mosaic_dim[1] * self.Ny))
        j_click = int(np.floor((x_click ) / self.mosaic_dim[2] * self.Nx))
        
        # Remove the label
        self.video_pages[self.current_page][i_click][j_click]['label'] = ''
        
        # Update the rectangles
        self.update_rectangles()


    def update_rectangles(self):
        '''Update the rectangles shown in the gui according to the labels'''
        # Reset rectangles
        self.rectangles = [[[] for _ in range(self.Nx)] for _ in range(self.Ny)]
        # Find the items labelled in the current page
        for i in range(self.Ny):
            for j in range(self.Nx):
                if self.video_pages[self.current_page][i][j]['label'] != '':
                    # Add the rectangle
                    p1 = (j*self.frame_dim[1], i*self.frame_dim[0])
                    p2 = ((j+1)*self.frame_dim[1], (i+1)*self.frame_dim[0])
                    label_text = self.video_pages[self.current_page][i][j]['label']
                    label_color = [bf['color'] for bf in self.labels if bf['name'] == label_text][0]
                    self.rectangles[i][j] = {'p1': p1, 'p2': p2, 
                                  'color': label_color, 'label': label_text}


    def main(self):
        # Settings
        videos_folder = r'G:\Videos'
        annotation_file = 'labels.json'
        status_file = 'status.json'
        video_ext = ['.mp4', '.avi']
        N_show_approx = 100
        screen_ratio = 16/9
        
        # Debug
        self.debug_verbose = 1
        
        # Find video files in the video folder
        videos_list = self.find_videos(videos_folder, video_ext)
        
        # Calculate number of videos per row/col
        self.Ny = int(np.sqrt(N_show_approx/screen_ratio))
        self.Nx = int(np.sqrt(N_show_approx*screen_ratio))
        
        # Calculate the video frame sizes
        cap = cv2.VideoCapture(videos_list[0])
        _, sample_frame = cap.read()
        self.frame_dim = sample_frame.shape
        cap.release()
        
        # Split the videos list into pages
        self.video_pages = self.list_to_pages(videos_list)
        self.current_page = 0
        
#        # Load status
#        if os.path.isfile(status_file):
#            with open(status_file, 'r') as jsonFile:
#                data = json.load(jsonFile)
#    
#            starting_day = data['day']
#            starting_video = data['last_video']
#            total_vid_ann = data['count']
#            
#            print('Status file found. Loading from day %s, video %s' %
#                  (starting_day, starting_video))
#        else:
#            starting_day = None
#            starting_video = None
#            total_vid_ann = 0
#            
#        # Load existing annotations
#        if os.path.isfile(annotation_file):
#            with open(annotation_file, 'r') as jsonFile:
#                self.annotations = json.load(jsonFile)
#        else:
#            self.annotations = []
        
        
        # Initialise the GUI
        cv2.namedWindow('sts_annotation')
        cv2.setMouseCallback('sts_annotation', self.click_callback)
        
        # Main loop
        run = True
        while run:
            # Get the mosaic for the current page
            videos_in_page = [item['video'] for sublist in self.video_pages[self.current_page] for item in sublist]
            mosaic, _ = self.create_mosaic(videos_in_page)
            self.mosaic_dim = mosaic.shape
            
            # Update the rectangles
            self.update_rectangles()
            
            # GUI loop
            run_this_page = True
            while run_this_page:
                for f in range(mosaic.shape[0]):
                    img = np.copy(mosaic[f, ...])
                    # Add rectangle to display selected sequence
                    rec_list = [item for sublist in self.rectangles for item in sublist
                                if item != []]
                    for rec in rec_list:
                        cv2.rectangle(img, rec['p1'], rec['p2'], rec['color'], 4)
                        textpt = (rec['p1'][0]+10, rec['p1'][1]+15)
                        cv2.putText(img, rec['label'], textpt, cv2.FONT_HERSHEY_SIMPLEX, 0.4, rec['color'])
                        
                    cv2.imshow('sts_annotation', img)
                    
                    key_input = cv2.waitKey(30)
                    if key_input == ord('n') or key_input == ord('N'):
                            self.current_page += 1
                            run_this_page = False
                            break
                        
                    if key_input == ord('b') or key_input == ord('B'):
                            self.current_page -= 1
                            run_this_page = False
                            break
                        
                    if key_input == ord('q') or key_input == ord('Q'):
                            run = None
                            run_this_page = False
                            break
            
            
        
#        self.rectangle = [[[] for _ in range(self.Nx)] for _ in range(self.Ny)]
        
        # Initialise the generator
#        generator = self.batch_generator(videos_folder, starting_day, starting_video, total_vid_ann)
        # Loop over the pages generated
#        for self.batch, self.video_names, self.status in generator:

            
#            # Save the status
#            if self.debug_verbose == 1:
#                print('Saving status...')
#            with open(status_file, 'w+') as jsonFile:
#                jsonFile.write(json.dumps(self.status, indent=1))
#                
#            # Backup of the annotations
#            if self.debug_verbose == 1:
#                print('Backing up annotations...')
#            if os.path.isfile(annotation_file):
#                copyfile(annotation_file, annotation_file+'.backup')
#                
#            # Save the annotations
#            if self.debug_verbose == 1:
#                print('Saving annotations...')
#            with open(annotation_file, 'w+') as jsonFile:
#                jsonFile.write(json.dumps(self.annotations, indent=1))
#    
            # Reset the rectangles
#            self.rectangle = [[[] for _ in range(self.Nx)] for _ in range(self.Ny)]
            
            # Exit the program
            if run is None:
                print('Quitting the program...')
                cv2.destroyAllWindows()
                return -1

           
if __name__ == '__main__':
    annotator = Annotator([
                {'name': 'walking', 
                'color': (0, 200, 0),
                'event': cv2.EVENT_LBUTTONDOWN},

                {'name': 'sitting', 
                'color': (0, 0, 200),
                'event': cv2.EVENT_LBUTTONDBLCLK},
                 
                 {'name': 'standing', 
                'color': (0, 140, 255),
                'event': cv2.EVENT_MBUTTONDOWN}
                ])

    annotator.main()