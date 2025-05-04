import torch
import cv2
from sort.sort import *
import pandas as pd
from classify_by_movement import *
import pandas as pd
from calculate_features import *
import os

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Load the YOLOv5 model from the checkpoint
model = torch.hub.load('ultralytics/yolov5', 'custom', path='../YOLO_model/best_yolov5x.pt')

# Video's frame rate
fps = 50  

# Path to images
#video_train_path = '../data/VISEM_Tracking/train/'
video_train_path = '../data/VISEM_Tracking/individual/'

# List all files and directories in the specified path
contents = os.listdir(video_train_path)
#print(contents)


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
    for idx,track in enumerate(tracks):
        xmin, ymin, xmax, ymax, track_id = track
        cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2  # Centroid
        tracking_data.append([frame_id, video_index, track_id, labels[idx], cx, cy, xmin, ymin, xmax, ymax])
        
        if track_id not in trajectories:
            trajectories[track_id] = []
        trajectories[track_id].append((cx, cy))
        
    return [trajectories,tracking_data]

def save_df(tracking_data):
    """
    Save data in csv file.
    
    Args:
        tracking_data: information about tracking in all videos
    """
    # Save tracking data to a CSV file
    print("save df")
    df = pd.DataFrame(tracking_data, columns=['frame_id', 'video_id', 'track_id', 'class', 'cx', 'cy', 'xmin', 'ymin', 'xmax', 'ymax'])
    df.to_csv('../results/data_sperm_tracking/sperm_tracking_data_12.csv', index=False)

def track_sperm():
    """
    Process to detect information about sperm for each video in train folder.
    """
    # Load tracker model
    tracker = Sort() 
    
    # Variables
    tracking_data = []
    trajectories = {}

    # Loop to analyse training videos
    for video_index in contents:
        video_path = '../data/VISEM_Tracking/train/' + video_index + '/' + video_index + '.mp4'
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        # Initialize frame id
        frame_id = 0
        
        # Process the video frame by frame
        while cap.isOpened():
            print("Frame_id:", frame_id)
            ret, frame = cap.read()
            if not ret:
                break

            # Run inference on the frame
            results = model(frame)
            
            # Extract bounding box information
            bbox_data = results.pandas().xyxy[0]
            detections = bbox_data[['xmin', 'ymin', 'xmax', 'ymax', 'confidence']].values
            labels = results.pandas().xyxy[0]['class'].values

            # Update the tracker with the detected bounding boxes
            tracks = tracker.update(detections)
            
            trajectories, tracking_data = update_trajectory(trajectories,tracking_data,tracks,labels,frame_id,video_index)
            
            # First 15 seconds
            if frame_id == 250:
                break
            
            frame_id += 1
        
        # Release the video capture object and close windows
        cap.release()
        save_df(tracking_data)
        
    #print(df)
    
    
if __name__ == "__main__":
    track_sperm()