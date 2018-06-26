# -*- coding: utf-8 -*-

import os
import json
import time
import threading
from shutil import copyfile
import numpy as np
import cv2

# BUG: error when showing the last page
# BUG: kill the background thread when quitting
# BUG: The time bar messes up with the click coordinates where the user clicks
# TODO: Review annotations
# TODO: Add check labels are changed
# TODO: Add check video file is a video file
# TODO: Add check for valid json files

class Annotator:
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
                        anno = [bf for bf in annotations if bf['video'] == videos_list[vid]]
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
        cached_page = self.current_page
        cold_start = True
        while self.run_thread:
            # The cached_page will be equal to self.current_page with a cold 
            # start, e self.current_page+1 if a page was already loaded. If
            # the user request a previous page (i.e. self.current_page-1),
            # the cached image should be discarded and a cold start should be
            # done.
            if cached_page not in {self.current_page, self.current_page+1}:
                if self.debug_verbose == 1:
                    print('(Thread) The cached page (%d) is different from the one requested (%d)' % 
                          (cached_page-1, self.current_page))
                cached_page = self.current_page
                cold_start = True
            
            # If cold_start is false, there already is an image in memory.
            # give it to main() and load the next one
            if not cold_start:
                # Set the mosaic to the last mosaic loaded
                self.mosaic = current_mosaic
                e_mosaic_ready.set()
                        
            # List video files
            current_mosaic = self.create_mosaic(cached_page)
            
            # Load the next page #
            if cached_page < len(self.video_pages)-1:
                cached_page += 1
            
            # Wait after loading two pages
            if cold_start:
                cold_start = False
            else:
                if self.debug_verbose == 1:
                    print('(Thread) In standby...')
                e_page_request.wait()
        
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
            print('(Thread) Page %d was correctly loaded' % page)
            
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
        i_click = int(np.floor((y_click) / self.mosaic_dim[1] * self.Ny))
        j_click = int(np.floor((x_click) / self.mosaic_dim[2] * self.Nx))
        i_click = np.min((np.max((0, i_click)), self.Ny-1))
        j_click = np.min((np.max((0, j_click)), self.Nx-1))
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
        bar = np.zeros((20, img.shape[1], 3))
        idt = int(fraction*img.shape[1])
        bar[:, 0:idt, 0] = color[0]
        bar[:, 0:idt, 1] = color[1]
        bar[:, 0:idt, 2] = color[2]
        img = np.concatenate((bar, img, bar), axis=0)
        return img


    def main(self):
        # Settings
        videos_folder = r'G:\STS_sequences\Videos'
        annotation_file = 'labels.json'
        status_file = 'status.json'
        video_ext = ['.mp4', '.avi']
        N_show_approx = 100
        screen_ratio = 16/9
        
        # Debug
        self.debug_verbose = 1
        
        # Calculate number of videos per row/col
        self.Ny = int(np.sqrt(N_show_approx/screen_ratio))
        self.Nx = int(np.sqrt(N_show_approx*screen_ratio))
        
        # Find video files in the video folder
        videos_list = self.find_videos(videos_folder, video_ext)
        # Calculate the video frame sizes
        cap = cv2.VideoCapture(videos_list[0])
        _, sample_frame = cap.read()
        self.frame_dim = sample_frame.shape
        cap.release()
 
       # Load existing annotations
        if os.path.isfile(annotation_file):
            with open(annotation_file, 'r') as json_file:
                existing_annotations = json.load(json_file)
        # Split the videos list into pages
        self.video_pages = self.list_to_pages(videos_list, existing_annotations)
        
        # Load status
        self.review_mode = False
        if os.path.isfile(status_file):
            with open(status_file, 'r') as json_file:
                data = json.load(json_file)
            
            # Load the status
            status_time = data['time']
            status_page = data['page']
            print('Status file found at %s. Loading from page %d' %
                  (status_time, status_page))
            
            # Set the status
            self.current_page = status_page
        else:
            # Start from page zero
            self.current_page = 0
 

        # Initialise the GUI
        cv2.namedWindow('MuViDat')
        cv2.setMouseCallback('MuViDat', self.click_callback)
        
        # Initialise threading events
        e_mosaic_ready = threading.Event()
        e_page_request = threading.Event()
        e_thread_off = threading.Event()
        self.run_thread = True
        tr = threading.Thread(target=self.mosaic_thread, 
                                         args=(e_mosaic_ready, e_page_request,
                                               e_thread_off))
        
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
            
            self.mosaic_dim = self.mosaic.shape
            
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
                        if self.current_page < len(self.video_pages):
                            self.current_page += 1
                            run_this_page = False
                            break
                        
                    if chr(key_input) in {'b', 'B'}:
                            if self.current_page > 0:
                                self.current_page -= 1
                                run_this_page = False
                                break
                        
                    if chr(key_input) in {'q', 'Q'}:
                        run = None
                        run_this_page = False
                        break
                        
                    if chr(key_input) in {'r', 'R'}:
                        existing_annotations = [item for page in self.video_pages for sublist in page for item in sublist if item['label']]
                        self.video_pages = self.list_to_pages(videos_list, existing_annotations, filter_label=True)
                        # Shut down the thread
                        self.run_thread = False
                        e_page_request.set()
                        e_thread_off.wait()
                        # Restart the thread and request page 0
                        self.run_thread = True
                        self.current_page = 0
                        e_mosaic_ready.clear()
                        e_page_request.set()
                        tr = threading.Thread(target=self.mosaic_thread, 
                                         args=(e_mosaic_ready, e_page_request,
                                               e_thread_off))
                        tr.start()
                        run_this_page = False
                        self.review_mode = True
                        break
            
            # Save the status
            if not self.review_mode:
                if self.debug_verbose == 1:
                    print('Saving status...')
                with open(status_file, 'w+') as json_file:
                    status = {'time': time.time(),
                              'page': self.current_page}
                    json_file.write(json.dumps(status, indent=1))

            # Backup of the annotations
            if self.debug_verbose == 1:
                print('Backing up annotations...')
            if os.path.isfile(annotation_file):
                copyfile(annotation_file, annotation_file+'.backup')

            # Save the annotations
            if self.debug_verbose == 1:
                print('Saving annotations...')
            with open(annotation_file, 'w+') as json_file:
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
    annotator = Annotator([
                {'name': 'sit down', 
                'color': (0, 1, 0),
                'event': cv2.EVENT_LBUTTONDOWN},

                {'name': 'stand up', 
                'color': (0, 0, 1),
                'event': cv2.EVENT_LBUTTONDBLCLK},
                 
                 {'name': 'ambiguous', 
                'color': (0, 1, 1),
                'event': cv2.EVENT_MBUTTONDOWN}
                ])

    annotator.main()