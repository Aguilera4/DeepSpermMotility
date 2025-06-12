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
    df.to_csv('../results/data_sperm_tracking/sperm_tracking_data_test_3.csv', index=False)
    

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
        frame, x, y, particle = track
        tracking_data.append([frame, video_index, particle, x, y, 0, 0, 0, 0, labels[idx]])
        
        '''if track_id not in trajectories:
            trajectories[track_id] = []
        trajectories[track_id].append((cx, cy))'''
        
    return [trajectories, tracking_data]


def track_sperm(model,list_videos,count_frames=250):
    """
    Process to detect information about sperm for each video in train folder.
    """
    
    # Variables
    image_width = 640
    image_height = 480

    # Loop to analyse training videos
    for video_index in list_videos:
        # Load tracker model 50, 20, 0.1
        tracker = Sort(max_age=50, min_hits=40, iou_threshold=0.0001)
        
        print(video_index)
        
        
        tracking_data = []
        trajectories = {}
        
        video_path = '../data/VISEM_Tracking/train/' + video_index + '/' + video_index + '.mp4'
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Process the video frame by frame
        for frame_id in range(0,total_frames):
            
            ret, frame = cap.read()
            if not ret:
                break
            
            
            #print(frame_id)
            results = pd.read_table('../data/VISEM_Tracking/train/' + video_index + '/labels/' + video_index + '_frame_' + str(frame_id) + '.txt', sep = ' ', names=['class', 'x_center', 'y_center', 'width', 'height'],header=None)

            detections = pd.DataFrame()
            
            xcenter = results['x_center'] * image_width
            ycenter = results['y_center'] * image_height
            __width = results['width'] * image_width
            __height = results['height'] * image_height
            
            detections['frame'] = np.full(len(results), frame_id)
            detections['xmin'] = xcenter
            detections['ymin'] = ycenter
            
            labels = results['class']
            
            # Update the tracker with the detected bounding boxes
            detections.columns = ["frame", "x", "y"]
            tracks = tp.link_df(detections, search_range=1,memory=4,adaptive_step=0.95,adaptive_stop=5,link_strategy='recursive')
            
            trajectories, tracking_data = update_trajectory(trajectories,tracking_data,tracks,labels,frame_id,video_index)
            
            for track_id, group in pd.DataFrame(tracking_data).groupby(2):
                print(group)
                for i in range(1,len(group)):
                    cv2.line(frame, (int(group.iloc[i-1,:][3]),int(group.iloc[i-1,:][4])), (int(group.iloc[i,:][3]),int(group.iloc[i,:][4])), (0,0,0), 1)
                
                
            # Display the frame
            cv2.imshow('Sperm Velocity', frame)
            
            if cv2.waitKey(int(1000 / 50)) & 0xFF == ord('q'):
                break
            
            
            # First 30 seconds
            if frame_id == count_frames:
                break
        
        
        df = pd.DataFrame(tracking_data, columns=['frame_id', 'video_id', 'track_id', 'cx', 'cy', 'xmin', 'ymin', 'xmax', 'ymax', 'class'])
        df.to_csv('../results/data_sperm_tracking/sperm_tracking_'+video_index+'_labels.csv', index=False)
        exit(0)
            
    #print(df)
    
    
if __name__ == "__main__":
    # Path to images
    #video_train_path = '../data/VISEM_Tracking/train/'
    video_train_path = '../data/VISEM_Tracking/train/'

    # List all files and directories in the specified path
    list_videos = os.listdir(video_train_path)
    print("Analyzed videos: ", list_videos)

    track_sperm("model",list_videos,1500)