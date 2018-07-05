# -*- coding: utf-8 -*-

import os
import json
import time
import threading
from shutil import copyfile
import numpy as np
import cv2

version_info = (0, 2, 1)
__version__ = '.'.join(str(c) for c in version_info)

# TODO: Add a function to jump to a specific page

class Annotator:
    '''Annotate multiple videos simultaneously by clicking on them.
    See demo.py for a working example.'''

    def __init__(self, labels, videos_folder, annotation_file='labels.json',
                 status_file='status.json', video_ext=['.mp4', '.avi'],
                 N_show_approx=100, screen_ratio=16/9, image_resize=1,
                 loop_duration=None):
        
        self.labels = labels
        
        # Settings
        self.videos_folder = videos_folder
        self.annotation_file = annotation_file
        self.status_file = status_file
        self.video_ext = video_ext
        self.N_show_approx = N_show_approx
        self.screen_ratio = screen_ratio
        self.image_resize = image_resize
        self.loop_duration = loop_duration

        # Hard coded settings
        self.timebar_h = 20  # Pixels
        self.rect_bord = 4  # Rectangle border
        # Debug
        self.debug_verbose = 0


    def video_to_clips(self, video_file, output_folder, resize=1, overlap=0, clip_length=90):
        '''Opens a long video file and saves it into several consecutive clips
        of predefined length'''
        # Initialise the counters
        clip_counter = 0
        video_frame_counter = 0
        # Generate clips path
        vid_name = os.path.splitext(os.path.basename(video_file))[0]
        clip_name = os.path.join(output_folder, '%s_clip_%%08d.mp4' % vid_name)
        # Calculate the overlap in number of frames
        assert 0 <= overlap < 1, 'The overlap must be in the range [0, 1['
        frames_overlap = int(clip_length*overlap)
        # Open the source video and read the framerate
        video_cap = cv2.VideoCapture(video_file)
        fps = video_cap.get(cv2.CAP_PROP_FPS)
        init = True
        while video_cap.isOpened():
            # Get the next video frame
            _, frame = video_cap.read()

            # Resize the frame
            if resize != 1 and frame is not None:
                frame = cv2.resize(frame, (0, 0), fx=resize, fy=resize)
                
            if frame is None:
                print('There was a problem processing frame %d' % video_frame_counter)
            
            # Initialise the video
            if init:
                frame_size = frame.shape
                video_length = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                clip_frame_counter = 0
                video_frame_counter = 0
                init = False
                clip_cap = cv2.VideoWriter(clip_name % clip_counter,
                                           fourcc, fps, 
                                           (frame_size[1], frame_size[0]))

            # Write the video frame into the clip
            clip_cap.write(frame)
            # Increase the index
            if clip_frame_counter < clip_length - 1:
                clip_frame_counter += 1
            else:
                # Save the complete clip
                print('\rClip %d complete (%.1f%%)' % (clip_counter,
                      video_frame_counter/video_length*100), end=' ')
                clip_cap.release()
                clip_frame_counter = 0
                clip_counter += 1
                # Initialise the next clip
                clip_cap = cv2.VideoWriter(clip_name % clip_counter,
                                           fourcc, fps, 
                                           (frame_size[1], frame_size[0]))
                # Set the next frame according to the overlap
                if overlap:
                    video_frame_counter -= frames_overlap
                    video_cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame_counter+1)
            
            # Interrupt when the videos is fully processed
            if video_frame_counter < video_length - 1:
                video_frame_counter += 1
            else:
                print('\rClip %d complete (100%%)' % clip_counter)
                clip_cap.release()
                video_cap.release()
                break

    
    def find_videos(self):
        '''Loop over the video folder looking for video files'''
        videos_list = []
        for folder, _, files in os.walk(self.videos_folder):
            for file in files:
                fullfile_path = os.path.join(folder, file)
                if os.path.splitext(fullfile_path)[1] in self.video_ext:
                    videos_list.append(os.path.join(folder, file))
                    
        return videos_list

    
    def build_dataset(self, videos_list, annotations):
        '''Creates the self.dataset array, containing a list of videos with the
        respective annotations'''
        N_videos = len(videos_list)
        self.dataset = [{'video': '', 'label': ''} for _ in range(N_videos)]
        print('Generating dataset array...')
        for vid in range(N_videos):
            # Add the video to the dataset
            self.dataset[vid]['video'] = videos_list[vid]
            # Add label to the dataset by checking that the realpath is the same
            real_path = os.path.realpath(videos_list[vid])
            anno = [bf for bf in annotations if bf['video'] == real_path]
            if anno:
                self.dataset[vid]['label'] = anno[0]['label']


    def build_pagination(self, filter_label=False):
        '''Take a list of videos in input and create a pagination array that
        splits the videos into pages'''
        # Filter the videos by labels if requested
        if filter_label:
            # TODO: This could be done in a more efficient way by preallocating pagination
            self.pagination = [[]]
            p = 0
            for vid in range(len(self.dataset)):                 
                # Add a new page
                if len(self.pagination[p]) == self.Nx*self.Ny:
                    self.pagination.append([])
                    p += 1
                    
                # Check if the video is labelled
                if self.dataset[vid]['label']:
                    self.pagination[p].append(vid)
                
            self.N_pages = p+1

        else:
            # Create the pagination
            self.N_pages = int(np.ceil(len(self.dataset)/(self.Nx*self.Ny)))
            self.pagination = [[] for _ in range(self.N_pages)]
            for vid in range(len(self.dataset)):
                p = int(np.floor(vid/(self.Nx*self.Ny)))
                self.pagination[p].append(vid)


    def mosaic_thread(self, e_mosaic_ready, e_page_request, e_thread_off):
        '''This function is a wrapper for create_mosaic that runs in a separate
        thread with main. When cold_start is true, it loads an image, returns 
        it to main, then load a new one in memory and finally wait. After this, 
        cold_start is set to false and at each successive call the function 
        simply returns the cached image, load the next one and waits.'''
        e_thread_off.clear()
        cold_start = True
        self.delete_cache = False
        while self.run_thread:
            # A cold_start is when no images are in memory. Simply load the current page
            if cold_start:
                # Get the mosaic of the current page
                current_mosaic = self.create_mosaic(self.current_page)
                page_in_cache = self.current_page
                cold_start = False

            # If the page in cache is the page requested, show it
            if not self.delete_cache and page_in_cache == self.current_page:
                self.mosaic = current_mosaic
                e_mosaic_ready.set()
            
                # Load the next mosaic
                next_page = self.current_page+self.page_direction
                next_page = int(np.max((0, np.min((next_page, self.N_pages-1)))))
                # Only load the next page if it's different from the current one
                if next_page != self.current_page:
                    current_mosaic = self.create_mosaic(next_page)
                    page_in_cache = next_page
            
                # Wait for the next page request
                e_page_request.wait()
            else:
                cold_start = True
                self.delete_cache = False
        
        if self.debug_verbose == 1:
            print('(Thread) The thread is dying now :(') 
            
        e_thread_off.set()


    def create_mosaic(self, page):
        '''This function loads videos and arrange them into a mosaic.'''
        # Select the videos from the pagination
        videos_list = [self.dataset[vid]['video'] for vid in self.pagination[page]]
        init = True
        i_scr, j_scr, k_time = 0, 0, 0
        # Loop over all the video files in the day folder
        for vi, video_file in enumerate(videos_list):
           
            # Deal with long lists
            if vi == self.Nx*self.Ny:
                print("The list of videos doesn't fit in the mosaic.")
                break
            
            # Open the video
            cap = cv2.VideoCapture(video_file)
            
            # Load the video frames
            while cap.isOpened():
                _, frame = cap.read()
                
                # Resize the frame
                if self.image_resize != 1 and frame is not None:
                    frame = cv2.resize(frame, (0, 0), fx=self.image_resize, fy=self.image_resize)
                
                # Initialise the mosaic         
                if init:
                    if frame is None:
                        raise Exception('The first video of the mosaic is invalid: %s.\n ' %
                                        video_file + 'Impossible to initialise the mosaic.') 
                    fdim = frame.shape
                    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    current_mosaic = np.zeros((n_frames, fdim[0]*self.Ny, fdim[1]*self.Nx, 3))
                    init = False
                
                # Check that the frame is valid
                if frame is not None and frame.shape == fdim:
                    # Add frame to the mosaic
                    current_mosaic[k_time, i_scr*fdim[0]:(i_scr+1)*fdim[0],
                                   j_scr*fdim[1]:(j_scr+1)*fdim[1], :] = frame[... , :]/255
                else:
                    # Show an image with an error message
                    print('Corrupted frame found at #%d of %s' % (k_time, video_file))
                    broken_frame = np.zeros(fdim)
                    pos = (10, fdim[0]//2)
                    cv2.putText(broken_frame, 'No frame #%d' % k_time,
                                pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                                (255, 255, 255), thickness=1)
                    current_mosaic[k_time, i_scr*fdim[0]:(i_scr+1)*fdim[0],
                               j_scr*fdim[1]:(j_scr+1)*fdim[1], :] = broken_frame

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
        
        # Set the label
        if event == cv2.EVENT_LBUTTONDOWN:
            label = self.labels[self.selected_label]
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
        
        # Convert i and j click into a single index
        vid_in_page = self.pagination[self.current_page]
        ind_click  = j_click*self.Ny + i_click
        self.dataset[vid_in_page[ind_click]]['label'] = label_text
        
        # Update the rectangles
        self.update_rectangles()


    def remove_label(self, x_click, y_click):
        '''Remove label from the annotations'''
        # Find the indices of the clicked sequence
        i_click, j_click = self.click_to_ij(x_click, y_click)
        
        # Convert i and j click into a single index
        vid_in_page = self.pagination[self.current_page]
        ind_click  = j_click*self.Ny + i_click
        self.dataset[vid_in_page[ind_click]]['label'] = ''
        
        # Update the rectangles
        self.update_rectangles()


    def update_rectangles(self):
        '''Update the rectangles shown in the gui according to the labels'''
        # Reset rectangles
        self.rectangles = []
        videos_list = [self.dataset[vid] for vid in self.pagination[self.current_page]]
        # Find the items labelled in the current page
        for vi, video in enumerate(videos_list):
            if not video['label']:
                continue
        
            # Convert vi into row and column
            j = int(np.floor(vi/self.Ny))
            i = int(np.mod(vi, self.Ny))
            # Add the rectangle
            hb = int(self.rect_bord/2)  # Half border
            p1 = (j*self.frame_dim[1] + hb, i*self.frame_dim[0] + hb)
            p2 = ((j+1)*self.frame_dim[1] - hb, (i+1)*self.frame_dim[0] - hb)
            label_text = video['label']
            label_color = [bf['color'] for bf in self.labels if bf['name'] == label_text][0]
            self.rectangles.append({'p1': p1, 'p2': p2, 
                          'color': label_color, 'label': label_text})


    def draw_anno_box(self, img):
        for rec in self.rectangles:
            cv2.rectangle(img, rec['p1'], rec['p2'], rec['color'], self.rect_bord)
            textpt = (rec['p1'][0]+10, rec['p1'][1]+15)
            cv2.putText(img, rec['label'], textpt, cv2.FONT_HERSHEY_SIMPLEX, 0.4, rec['color'])


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
        if not os.path.isfile(self.annotation_file):
            print('No annotation found at %s' % self.annotation_file)
            return []
            
        with open(self.annotation_file, 'r') as json_file:
            try:
                annotations = json.load(json_file)
                print('Existing annotation found: %d items' % len(annotations))
            except json.JSONDecodeError:
                print('Unable to load annotations from %s' % self.annotation_file)
                return []
            
        # Check for absolute/relative paths of annotated videos and
        # make sure that labels are valid
        valid_labels = {bf['name'] for bf in self.labels}
        for anno in annotations:
            # If the annotation has a relative path, it is relative to the 
            # annotation file's folder
            if not os.path.isabs(anno['video']):
                anno['video'] = os.path.join(os.path.dirname(self.annotation_file), anno['video'])
            
            # Resolve path to allow future string comparison
            anno['video'] = os.path.realpath(anno['video'])
            
            # Check if the label is part of the valid set
            if anno['label'] and anno['label'] not in valid_labels:
                print(('Found label "%s" in %s, not compatible with %s. ' +
                      'All the labels will be discarded') %
                      (anno['label'], self.annotation_file, valid_labels))
                return []
            
        return annotations


    def show_label_guide(self):
        '''Show the labels available with the keyboard key to select them'''
        print('\n' + '-'*80)
        print('Please press a number key to select a label and use left/right '
              'click to add/remove labels')
        print('Labels available:')
        for li, label in enumerate(self.labels):
            print(' - %d: %s' % (li+1, label['name']))
        print('-'*80 + '\n')


    def load_status(self):
        '''Load the status from self.status_file and set self.current_page'''
        if os.path.isfile(self.status_file):
            with open(self.status_file, 'r') as json_file:
                try:
                    data = json.load(json_file)
                    # Load the status
                    status_time = data['time']
                    status_vid = data['first_video_id']
                    print('Status file found at %s' % time.ctime(status_time))
                except json.JSONDecodeError:
                    status_vid = 0
                    print('Error while loading the status file.')
            
            # Find the page of the video saved in the status
            for p in range(len(self.pagination)):
                if status_vid in self.pagination[p]:
                    self.current_page = p
                    break
            else:
                print(''''Status file belongs to a different session. Starting
                      form page 0''')
                self.current_page = 0
                
            self.current_page = int(np.max((0, np.min((self.current_page, self.N_pages-1)))))
        else:
            # Start from page zero
            self.current_page = 0
            
    
    def save_annotations(self):
        '''Save the annotations into a json file'''
        # Backup of the annotations first
        if self.debug_verbose == 1:
            print('Backing up annotations...')
        if os.path.isfile(self.annotation_file):
            copyfile(self.annotation_file, self.annotation_file+'.backup')

        # Save the annotations
        if self.debug_verbose == 1:
            print('Saving annotations...')
        with open(self.annotation_file, 'w+') as json_file:
            # Save non empty labels only
            non_empty = [item for item in self.dataset if item['label']]
            json_file.write(json.dumps(non_empty, indent=1))

    
    def save_status(self):
        '''Save the status into a json file'''
        # Save the status
        if not self.review_mode:
            if self.debug_verbose == 1:
                print('Saving status...')
            with open(self.status_file, 'w+') as json_file:
                status = {'time': time.time(),
                          'first_video_id': self.pagination[self.current_page][0]}
                json_file.write(json.dumps(status, indent=1))


    def process_keyboard_input(self, key_input, run):
        '''Deal with the user keyboard input'''
        run_this_page = True
        
        # Next page
        if chr(key_input) in {'n', 'N'}:
            if self.current_page < self.N_pages-1:
                self.current_page += 1
                self.page_direction = +1
                run_this_page = False
                
        # Previous page
        if chr(key_input) in {'b', 'B'}:
            if self.current_page > 0:
                self.current_page -= 1
                self.page_direction = -1
                run_this_page = False
                
        # Select label
        if chr(key_input) in {chr(d) for d in range(ord('0'),ord('9')+1)}:

            if int(chr(key_input)) > len(self.labels):
                print('Error: label %s not implemented' % chr(key_input))
            else:
                self.selected_label = int(chr(key_input))-1
                print('Label selected: %s' % self.labels[self.selected_label]['name'])
        
        # Reviewing mode
        if chr(key_input) in {'r', 'R'}:
            # Check if review_mode is active
            if self.review_mode:
                # Exit review mode
                self.current_page = self.remember_page
                self.build_pagination(filter_label=False)
                self.review_mode = False
                self.delete_cache = True
                run_this_page = False
            else:
                # Update the pagination using labelled videos only
                self.build_pagination(filter_label=True)
                print('Entering reviewing mode. Press "r" again to quit')
                self.remember_page = self.current_page
                self.current_page = 0
                self.review_mode = True
                self.delete_cache = True
                run_this_page = False

        # Quit
        if chr(key_input) in {'q', 'Q'}:
            run = None
            run_this_page = False
                
        return run_this_page, run


    def main(self):
        # Find video files in the video folder
        videos_list = self.find_videos()
        if not videos_list:
            print('No videos found at %s' % self.videos_folder)
            return -1
        
        # Calculate the video frame sizes and loop duration
        cap = cv2.VideoCapture(videos_list[0])
        if self.loop_duration:
            # Loop duration defined by the user
            n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            delay = int(self.loop_duration*1000/n_frames)     
        else:
            # Automatic loop duration based on fps
            delay = int(1000/cap.get(cv2.CAP_PROP_FPS))
            
        _, sample_frame = cap.read()
        self.frame_dim = [int(bf*self.image_resize) for bf in sample_frame.shape]
        cap.release()
        
        # Calculate number of videos per row/col
        self.Ny = int(np.sqrt(self.N_show_approx/self.screen_ratio * self.frame_dim[1]/self.frame_dim[0]))
        self.Nx = int(np.sqrt(self.N_show_approx*self.screen_ratio * self.frame_dim[0]/self.frame_dim[1]))
 
       # Load existing annotations and build pagination
        existing_annotations = self.load_annotations()
        self.build_dataset(videos_list, existing_annotations)
        self.build_pagination()
                
        # Load status
        self.review_mode = False  # In review mode, the status is not saved
        self.load_status()
        self.page_direction = +1  # Used for the cache preload

        # Initialise the GUI
        self.show_label_guide()
        self.selected_label = 0
        cv2.namedWindow('MuViLab')
        cv2.setMouseCallback('MuViLab', self.click_callback)
        
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
            
            print('\rShowing page %d/%d' % (self.current_page+1, self.N_pages), end=' ')
            
            # GUI loop
            run_this_page = True
            while run_this_page:
                for f in range(self.mosaic.shape[0]):
                    tic = time.time()
                    img = np.copy(self.mosaic[f, ...])
                    # Draw annotation box and timebar
                    self.draw_anno_box(img)
                    img = self.add_timebar(img, f/self.mosaic.shape[0])
                    
                    cv2.imshow('MuViLab', img)
                    
                    # Deal with the keyboard input
                    toc = int((time.time()-tic)*1000)
                    wait = int(np.max((1, delay-toc)))
                    key_input = cv2.waitKey(wait)
                    if key_input == -1:
                        continue
                    run_this_page, run = self.process_keyboard_input(key_input, run)
                    if not run_this_page:
                        break
            
            # Save status and annotations
            self.save_status()
            self.save_annotations()
            
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
    videos_folder = r'G:\STS_sequences\Videos'
    labels = [{'name': 'sit down', 'color': (0, 1, 0)},
              {'name': 'stand up', 'color': (0, 0, 1)},                 
              {'name': 'ambiguous', 'color': (0, 1, 1)}]
    annotator = Annotator(labels, videos_folder, annotation_file=r"G:\STS_sequences\labels.json")
    annotator.main()