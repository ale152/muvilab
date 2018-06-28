# -*- coding: utf-8 -*-

import os
import json
import time
import threading
from shutil import copyfile
import numpy as np
import cv2

# TODO: Add a preprocessing function that splits long videos into clips (like the demo)
# TODO: Status should save the first video in the page shown, rather than the page number!
# TODO: label using numbers on the keyboard. Press 1 and every click will be labelled as 1
# TODO: Add check video file is a video file

class Annotator:
    '''Annotate multiple videos simultaneously by clicking on them.
    See demo.py for a working example.'''

    def __init__(self, labels, videos_folder, annotation_file='labels.json',
                 status_file='status.json', video_ext=['.mp4', '.avi'],
                 N_show_approx=100, screen_ratio=16/9):
        
        self.labels = labels
        
        # Settings
        self.videos_folder = videos_folder
        self.annotation_file = annotation_file
        self.status_file = status_file
        self.video_ext = video_ext
        self.N_show_approx = N_show_approx
        self.screen_ratio = screen_ratio

        # Hard coded settings
        self.timebar_h = 20  # Pixels
        
        # Debug
        self.debug_verbose = 0

    
    def find_videos(self):
        '''Loop over the video folder looking for video files'''
        videos_list = []
        for folder, _, files in os.walk(self.videos_folder):
            for file in files:
                fullfile_path = os.path.join(folder, file)
                if os.path.splitext(fullfile_path)[1] in self.video_ext:
                    videos_list.append(os.path.join(folder, file))
                    
        return videos_list

    
    def list_to_pages(self, videos_list, annotations, filter_label=False):
        '''Split a list of videos into an array arranged by pages of mosaics'''
        # Filter the videos by labels if requested
        if filter_label:
            videos_list = [bf['video'] for bf in annotations]

        # Convert the list into a list of pages of grids
        N_pages = int(np.ceil(len(videos_list)/self.Nx/self.Ny))
        video_pages = [[[{'video': '', 'label': ''} for _ in range(self.Ny)] for _ in range(self.Nx)] for _ in range(N_pages)]
        vid = 0
        for p in range(N_pages):
            for j in range(self.Nx):
                for i in range(self.Ny):
                    if vid < len(videos_list):
                        # Add the video to the grid
                        video_pages[p][j][i]['video'] = videos_list[vid]
                        # Add the annotation to the grid
                        anno = [bf for bf in annotations if os.path.samefile(bf['video'], videos_list[vid])]
                        if anno:
                            video_pages[p][j][i]['label'] = anno[0]['label']
                        # Go to the next element in the video_list
                        vid += 1
        
        return video_pages


    def mosaic_thread(self, e_mosaic_ready, e_page_request, e_thread_off):
        '''This function is a wrapper for create_mosaic that runs in a separate
        thread with main. When cold_start is true, it loads an image, returns 
        it to main, then load a new one in memory and finally wait. After this, 
        cold_start is set to false and at each successive call the function 
        simply returns the cached image, load the next one and waits.'''
        e_thread_off.clear()
        cold_start = True
        while self.run_thread:
            # A cold_start is when no images are in memory. Simply load the current page
            if cold_start:
                # Get the mosaic of the current page
                current_mosaic = self.create_mosaic(self.current_page)
                page_in_cache = self.current_page
                cold_start = False

            # If the page in cache is the page requested, show it
            if page_in_cache == self.current_page:
                self.mosaic = current_mosaic
                e_mosaic_ready.set()
            
                # Load the next mosaic
                next_page = self.current_page+self.page_direction
                next_page = int(np.max((0, np.min((next_page, len(self.video_pages)-1)))))
                # Only load the next page if it's different from the current one
                if next_page != self.current_page:
                    current_mosaic = self.create_mosaic(next_page)
                    page_in_cache = next_page
            
                # Wait for the next page request
                e_page_request.wait()
            else:
                cold_start = True
        
        if self.debug_verbose == 1:
            print('(Thread) The thread is dying now :(') 
            
        e_thread_off.set()


    def create_mosaic(self, page):
        '''This function loads videos and arrange them into a mosaic. The videos
        are taken from self.video_pages, the input argument page'''
        videos_list = [item['video'] for sublist in self.video_pages[page] for item in sublist]
        init = True
        # Loop over all the video files in the day folder
        for vi, video_file in enumerate(videos_list):
            #print('\rLoading file %s' % video_file, end=' ')
            
            # Deal with long lists
            if vi == self.Nx*self.Ny:
                print("The list of videos doesn't fit in the mosaic.")
                break
            
            # Open the video
            cap = cv2.VideoCapture(video_file)
            
            # Load the video frames
            while cap.isOpened():
                ret, frame = cap.read()
                
                # Initialise the video            
                if init:
                    fdim = frame.shape
                    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    current_mosaic = np.zeros((n_frames, fdim[0]*self.Ny, fdim[1]*self.Nx, 3))
                    i_scr, j_scr, k_time = 0, 0, 0
                    init = False
                
                # Add video to the grid
                current_mosaic[k_time, i_scr*fdim[0]:(i_scr+1)*fdim[0],
                             j_scr*fdim[1]:(j_scr+1)*fdim[1], :] = frame[... , :]/255
                
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
                
        if self.debug_verbose == 1:
            print('(Thread) Mosaic for page %d was correctly created' % page)
            
        return current_mosaic


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


    def click_to_ij(self, x_click, y_click):
        '''Convert the x-y coordinates of the mouse into i-j elements of the 
        mosaic'''
        i_click = int(np.floor((y_click-self.timebar_h) / self.mosaic.shape[1] * self.Ny))
        j_click = int(np.floor((x_click) / self.mosaic.shape[2] * self.Nx))
        i_click = int(np.min((np.max((0, i_click)), self.Ny-1)))
        j_click = int(np.min((np.max((0, j_click)), self.Nx-1)))
        return i_click, j_click

    def set_label(self, label_text, label_color, x_click, y_click):
        '''Set a specific label based on the user click input'''
        # Find the indices of the clicked sequence
        i_click, j_click = self.click_to_ij(x_click, y_click)
        
        # Create the label
        self.video_pages[self.current_page][j_click][i_click]['label'] = label_text
        
        # Update the rectangles
        self.update_rectangles()


    def remove_label(self, x_click, y_click):
        '''Remove label from the annotations'''
        # Find the indices of the clicked sequence
        i_click, j_click = self.click_to_ij(x_click, y_click)
        
        # Remove the label
        self.video_pages[self.current_page][j_click][i_click]['label'] = ''
        
        # Update the rectangles
        self.update_rectangles()


    def update_rectangles(self):
        '''Update the rectangles shown in the gui according to the labels'''
        # Reset rectangles
        self.rectangles = [[[] for _ in range(self.Ny)] for _ in range(self.Nx)]
        # Find the items labelled in the current page
        for j in range(self.Nx):
            for i in range(self.Ny):
                if not self.video_pages[self.current_page][j][i]['label']:
                    continue
            
                # Add the rectangle
                p1 = (j*self.frame_dim[1], i*self.frame_dim[0])
                p2 = ((j+1)*self.frame_dim[1], (i+1)*self.frame_dim[0])
                label_text = self.video_pages[self.current_page][j][i]['label']
                label_color = [bf['color'] for bf in self.labels if bf['name'] == label_text][0]
                self.rectangles[j][i] = {'p1': p1, 'p2': p2, 
                              'color': label_color, 'label': label_text}

    
    def add_timebar(self, img, fraction, color=(0.2, 0.5, 1)):
        '''Add a timebar on the image'''
        bar = np.zeros((self.timebar_h, img.shape[1], 3))
        idt = int(fraction*img.shape[1])
        bar[:, 0:idt, 0] = color[0]
        bar[:, 0:idt, 1] = color[1]
        bar[:, 0:idt, 2] = color[2]
        img = np.concatenate((bar, img, bar), axis=0)
        return img


    def load_annotations(self):
        '''Load annotations from self.annotation_file'''
        if os.path.isfile(self.annotation_file):
            with open(self.annotation_file, 'r') as json_file:
                try:
                    annotations = json.load(json_file)
                    print('Existing annotation found: %d items' % len(annotations))
                    # Check for absolute/relative paths of annotated videos and
                    # make sure that labels are valid
                    valid_labels = {bf['name'] for bf in self.labels}
                    for anno in annotations:
                        # If the annotation has a relative path, it is relative
                        # to the annotation file's folder
                        if not os.path.isabs(anno['video']):
                            anno['video'] = os.path.join(os.path.dirname(self.annotation_file), anno['video'])
                        
                        # Check if the label is part of the valid set
                        if anno['label'] and anno['label'] not in valid_labels:
                            print(('Found label "%s" in %s, not compatible with %s. ' +
                                  'All the labels will be discarded') %
                                  (anno['label'], self.annotation_file, valid_labels))
                            annotations = []
                            break
                    
                    
                except json.JSONDecodeError:
                    print('Unable to load annotations from %s' % self.annotation_file)
                    annotations = []
        else:
            print('No annotation found at %s' % self.annotation_file)
            annotations = []
            
        return annotations


    def load_status(self):
        '''Load the status from self.status_file and set self.current_page'''
        if os.path.isfile(self.status_file):
            with open(self.status_file, 'r') as json_file:
                try:
                    data = json.load(json_file)
                    # Load the status
                    status_time = data['time']
                    status_page = data['page']
                    print('Status file found at %s. Loading from page %d' %
                          (status_time, status_page))
                except json.JSONDecodeError:
                    status_page = 0
                    print('Error while loading the status file.')
            
            # Set the status
            self.current_page = status_page
            self.current_page = int(np.max((0, np.min((self.current_page, len(self.video_pages)-1)))))
        else:
            # Start from page zero
            self.current_page = 0


    def main(self):
        # Find video files in the video folder
        videos_list = self.find_videos()
        if not videos_list:
            print('No videos found at %s' % self.videos_folder)
            return -1
        
        # Calculate the video frame sizes
        cap = cv2.VideoCapture(videos_list[0])
        _, sample_frame = cap.read()
        self.frame_dim = sample_frame.shape
        cap.release()
        
        # Calculate number of videos per row/col
        self.Ny = int(np.sqrt(self.N_show_approx/self.screen_ratio * self.frame_dim[1]/self.frame_dim[0]))
        self.Nx = int(np.sqrt(self.N_show_approx*self.screen_ratio * self.frame_dim[0]/self.frame_dim[1]))
 
       # Load existing annotations
        existing_annotations = self.load_annotations()

        # Split the videos list into pages
        self.video_pages = self.list_to_pages(videos_list, existing_annotations)
        
        # Load status
        self.review_mode = False  # In review mode, the status is not saved
        self.load_status()
 
        # Page direction (used for the cache)
        self.page_direction = +1

        # Initialise the GUI
        cv2.namedWindow('MuViDat')
        cv2.setMouseCallback('MuViDat', self.click_callback)
        
        # Define events and thread
        e_mosaic_ready = threading.Event()  # Tells the main when the mosaic is ready to be shown
        e_page_request = threading.Event()  # Tells the thread that a new mosaic has been requested
        e_thread_off = threading.Event()  # Tells the main that the thread is done
        self.run_thread = True
        tr = threading.Thread(target=self.mosaic_thread, 
                              args=(e_mosaic_ready, e_page_request,
                                    e_thread_off))
        # Initialise the events
        e_mosaic_ready.clear()
        e_page_request.set()
        tr.start()

        if self.debug_verbose == 1:
            print('(Main) Mosaic generator started in background, waiting for the mosaic...')
        
        # Main loop
        run = True
        while run:
            # Wait for the mosaic to be generated
            if self.debug_verbose == 1:
                print('Main is waiting for the mosaic...')
            
            e_mosaic_ready.wait()  # Wait for the mosaic
            e_page_request.clear()  # Tell the thread to wait for a page request
            
            if self.debug_verbose == 1:
                print('(Main) Mosaic received in the main loop')
            
            # Update the rectangles
            self.update_rectangles()
            
            print('Showing page %d of %d' % (self.current_page, len(self.video_pages)))
            
            # GUI loop
            run_this_page = True
            while run_this_page:
                for f in range(self.mosaic.shape[0]):
                    img = np.copy(self.mosaic[f, ...])
                    # Add rectangle to display selected sequence
                    rec_list = [item for sublist in self.rectangles for item in sublist if item]
                    for rec in rec_list:
                        cv2.rectangle(img, rec['p1'], rec['p2'], rec['color'], 4)
                        textpt = (rec['p1'][0]+10, rec['p1'][1]+15)
                        cv2.putText(img, rec['label'], textpt, cv2.FONT_HERSHEY_SIMPLEX, 0.4, rec['color'])
                    
                    # Add a timebar
                    img = self.add_timebar(img, f/self.mosaic.shape[0])
                    
                    cv2.imshow('MuViDat', img)
                    
                    # Deal with the keyboard input
                    key_input = cv2.waitKey(30)
                    if key_input == -1:
                        continue
                    
                    if chr(key_input) in {'n', 'N'}:
                        if self.current_page < len(self.video_pages)-1:
                            self.current_page += 1
                            self.page_direction = +1
                            run_this_page = False
                            break
                        
                    if chr(key_input) in {'b', 'B'}:
                            if self.current_page > 0:
                                self.current_page -= 1
                                self.page_direction = -1
                                run_this_page = False
                                break
                        
                    if chr(key_input) in {'q', 'Q'}:
                        run = None
                        run_this_page = False
                        break
                        
                    if chr(key_input) in {'r', 'R'}:
                        # Update self.video_pages using labelled data only
                        existing_annotations = [item for page in self.video_pages for sublist in page for item in sublist if item['label']]
                        if not existing_annotations:
                            # No annotations, no reviewing mode
                            print('Please annotate some videos before entering reviewing mode')
                            continue
                        
                        print('Entering reviewing mode. Press "q" to quit')
                        self.video_pages = self.list_to_pages(videos_list, existing_annotations, filter_label=True)
                        self.current_page = 0
                        self.review_mode = True
                        run_this_page = False
                        break
            
            # Save the status
            if not self.review_mode:
                if self.debug_verbose == 1:
                    print('Saving status...')
                with open(self.status_file, 'w+') as json_file:
                    status = {'time': time.time(),
                              'page': self.current_page}
                    json_file.write(json.dumps(status, indent=1))

            # Backup of the annotations
            if self.debug_verbose == 1:
                print('Backing up annotations...')
            if os.path.isfile(self.annotation_file):
                copyfile(self.annotation_file, self.annotation_file+'.backup')

            # Save the annotations
            if self.debug_verbose == 1:
                print('Saving annotations...')
            with open(self.annotation_file, 'w+') as json_file:
                # Save non empty labels only
                non_empty = [item for page in self.video_pages for sublist in page for item in sublist if item['label']]
                json_file.write(json.dumps(non_empty, indent=1))
            
            # Exit the program
            if run is None:
                print('Quitting the program...')
                cv2.destroyAllWindows()
                self.run_thread = False
                e_page_request.set()
                return -1
            
            # Ask the mosaic generator for the next page
            if self.debug_verbose == 1:
                print('(Main) New mosaic requested, waiting for it')
            e_mosaic_ready.clear()  # Set the mosaic to not ready
            e_page_request.set()  # Request a new mosaic

           
if __name__ == '__main__':
    videos_folder = r'./Videos'
    labels = [{'name': 'sit down', 
                'color': (0, 1, 0),
                'event': cv2.EVENT_LBUTTONDOWN},

                {'name': 'stand up', 
                'color': (0, 0, 1),
                'event': cv2.EVENT_LBUTTONDBLCLK},
                 
                 {'name': 'ambiguous', 
                'color': (0, 1, 1),
                'event': cv2.EVENT_MBUTTONDOWN}]
    annotator = Annotator(labels, videos_folder)
    annotator.main()