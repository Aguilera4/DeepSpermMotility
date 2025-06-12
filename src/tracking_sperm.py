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
    df.to_csv('../results/data_sperm_tracking/sperm_tracking_data_12_30s_15_20_01.csv', index=False)
    

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

    # Loop to analyse training videos
    for video_index in list_videos:
        video_path = '../data/VISEM_Tracking/train/' + video_index + '/' + video_index + '.mp4'
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        # Load tracker model 50, 20, 0.1
        tracker = Sort(max_age=15, min_hits=20, iou_threshold=0.1)
        
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
            
            '''for row in tracks:
                cv2.rectangle(frame, (int(row[0]), int(row[1])), (int(row[2]), int(row[3])), (0,0,0), 1)
            
            
            for track_id, group in pd.DataFrame(tracking_data).groupby(2):

                for i in range(1,len(group)):
                    cv2.line(frame, (int(group.iloc[i-1,:][3]),int(group.iloc[i-1,:][4])), (int(group.iloc[i,:][3]),int(group.iloc[i,:][4])), (0,0,0), 1)
                
            for key, value in trajectories.items():
                for i in range(1,len(value)):
                    cv2.line(frame, (int(value[i-1][0]),int(value[i-1][1])), (int(value[i][0]),int(value[i][1])), (0,0,0), 1)
                    
                    
                    
            # Display the frame
            cv2.imshow('Sperm Velocity', frame)
            
            if cv2.waitKey(int(1000 / 50)) & 0xFF == ord('q'):
                break
            '''
            
            # First 30 seconds
            if frame_id == count_frames:
                break
            
            frame_id += 1
        
        # Release the video capture object and close windows
        cap.release()
        #cv2.destroyAllWindows()
        save_df(tracking_data)
        
    #print(df)
    
    
if __name__ == "__main__":
    
    # Load the YOLOv5 model from the checkpoint
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='../YOLO_model/best_yolov5x.pt')

    # Path to images
    #video_train_path = '../data/VISEM_Tracking/train/'
    video_train_path = '../data/VISEM_Tracking/individual/'

    # List all files and directories in the specified path
    list_videos = os.listdir(video_train_path)
    print("Analyzed videos: ", list_videos)

    track_sperm(model,list_videos,1500)