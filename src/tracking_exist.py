import torch
import cv2
from sort.sort import *
import pandas as pd
from classify_by_movement import *
import pandas as pd
from functions_features import *
import os
import trackpy as tp

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
    df.to_csv('../results/data_sperm_tracking/sperm_tracking_data_test_track.csv', index=False)
    

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
    for idx, track in tracks.iterrows():
        xmin = track['xmin']
        xmax = track['xmax']
        ymin = track['ymin']
        ymax = track['ymax']
        cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2  # Centroid
        tracking_data.append([frame_id, video_index, track['track_id'], cx, cy, xmin, ymin, xmax, ymax, labels[idx]])
        
    return [trajectories, tracking_data]


def track_sperm(model,list_videos,count_frames=250):
    """
    Process to detect information about sperm for each video in train folder.
    """
    
    # Variables
    image_width = 640
    image_height = 480
    
    
        
    tracking_data = []
    trajectories = {}

    # Loop to analyse training videos
    for video_index in list_videos:
        
        print(video_index)
        
        video_path = '../data/VISEM_Tracking/train/' + video_index + '/' + video_index + '.mp4'
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Process the video frame by frame
        for frame_id in range(0,total_frames):
            #print(frame_id)
            
            if os.path.exists('../VISEM-Tracking/VISEM_Tracking_Train_v4/Train/' + video_index + '/labels_ftid/' + video_index + '_frame_' + str(frame_id) + '_with_ftid.txt'):
                
                results = pd.read_table('../VISEM-Tracking/VISEM_Tracking_Train_v4/Train/' + video_index + '/labels_ftid/' + video_index + '_frame_' + str(frame_id) + '_with_ftid.txt', sep = ' ', names=['track_id','class', 'x_center', 'y_center', 'width', 'height'],header=None)

                detections = pd.DataFrame()
                
                xcenter = results['x_center'] * image_width
                ycenter = results['y_center'] * image_height
                __width = results['width'] * image_width
                __height = results['height'] * image_height
                
                
                detections['track_id'] = results['track_id']
                detections['xmin'] = xcenter - (__width/2)
                detections['ymin'] = ycenter - (__height/2)
                detections['xmax'] = xcenter + (__width/2)
                detections['ymax'] = ycenter + (__height/2)
                detections['class'] = results['class']
                
                labels = results['class']
                
                trajectories, tracking_data = update_trajectory(trajectories,tracking_data,detections,labels,frame_id,video_index)
            
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