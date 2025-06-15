import torch
import cv2
from sort.sort import *
import pandas as pd
from classify_by_movement import *
import pandas as pd
from functions_features import *
import os

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

def save_df(tracking_data):
    """
    Save data in csv file.
    
    Args:
        tracking_data: information about tracking in all videos
    """
    # Save tracking data to a CSV file
    print("save df")
    df = pd.DataFrame(tracking_data, columns=['frame_id', 'video_id', 'track_id', 'cx', 'cy', 'xmin', 'ymin', 'xmax', 'ymax', 'class'])
    df.to_csv('../results/data_sperm_tracking/tracking_30s.csv', index=False)
    

def update_trajectory(trajectories,tracking_data,tracks,labels,frame_id,video_index):
    """
    Update the trajectory.
    
    Args:
        trajectories: trajectories
        tracking_data: information about tracking in all videos
        tracks: tracks of curent frame
        labels: labels of sperms
        frame_id: number of frame
        video_index: identifier of video
        
    Returns:
        list: trajectories.
        list: information about the sperms tracking in frame.
    """
    # Inside the frame processing loop:
    for idx, track in enumerate(tracks):
        xmin, ymin, xmax, ymax, track_id = track
        cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2  # Centroid
        tracking_data.append([frame_id, video_index, track_id, cx, cy, xmin, ymin, xmax, ymax, labels[idx]])
        
        '''if track_id not in trajectories:
            trajectories[track_id] = []
        trajectories[track_id].append((cx, cy))'''
        
    return [trajectories, tracking_data]


def track_sperm(model,list_videos,count_frames=250):
    """
    Process to detect information about sperm for each video in train folder.
    """
    
    # Variables
    tracking_data = []
    trajectories = {}
    image_width = 640
    image_height = 480

    # Loop to analyse training videos
    for video_index in list_videos:
        # Load tracker model 50, 20, 0.1
        tracker = Sort(max_age=15, min_hits=20, iou_threshold=0.1)
        
        print(video_index)
        
        video_path = '../data/VISEM_Tracking/train/' + video_index + '/' + video_index + '.mp4'
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Process the video frame by frame
        for frame_id in range(0,total_frames):
            if os.path.exists('../data/VISEM_Tracking/train/' + video_index + '/labels/' + video_index + '_frame_' + str(frame_id) + '.txt'):
                results = pd.read_table('../data/VISEM_Tracking/train/' + video_index + '/labels/' + video_index + '_frame_' + str(frame_id) + '.txt', sep = ' ', names=['class', 'x_center', 'y_center', 'width', 'height'],header=None)

                detections = pd.DataFrame()
                
                xcenter = results['x_center'] * image_width
                ycenter = results['y_center'] * image_height
                __width = results['width'] * image_width
                __height = results['height'] * image_height
                
                detections['xmin'] = xcenter - (__width/2)
                detections['ymin'] = ycenter - (__height/2)
                detections['xmax'] = xcenter + (__width/2)
                detections['ymax'] = ycenter + (__height/2)
                detections['class'] = results['class'] * 0
                
                labels = results['class']
                
                # Update the tracker with the detected bounding boxes
                tracks = tracker.update(detections.to_numpy())
                
                trajectories, tracking_data = update_trajectory(trajectories,tracking_data,tracks,labels,frame_id,video_index)
            
            # First 30 seconds
            if frame_id == count_frames:
                break
        
        save_df(tracking_data)
            
    #print(df)
    
    
if __name__ == "__main__":
    # Path to images
    #video_train_path = '../data/VISEM_Tracking/train/'
    video_train_path = '../data/VISEM_Tracking/train/'

    # List all files and directories in the specified path
    list_videos = os.listdir(video_train_path)
    print("Analyzed videos: ", list_videos)

    track_sperm("model",list_videos,1500)