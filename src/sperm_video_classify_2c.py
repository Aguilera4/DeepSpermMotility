import seaborn as sns
from sort.sort import *
import pandas as pd
import matplotlib.pyplot as plt
from classify_by_movement import *
import pandas as pd
from functions_features import *
import cv2
import torch
from sort.sort import *
import joblib
from preprocessing import *

matplotlib.use("TkAgg")  # Use Tkinter-based backend

import warnings
warnings.filterwarnings("ignore")

fps = 50

def traking_video(video_path,name_video):# Video capture
    # Load the YOLOv5 model from the checkpoint
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='../YOLO_model/best_yolov5x.pt')
    
    cap = cv2.VideoCapture(video_path)

    # Initialize the tracking algorithm
    tracker = Sort(max_age=15, min_hits=20, iou_threshold=0.1)

    # Initialize variables
    tracking_data = []
    frame_id = 0

    # Process the video frame by frame
    while cap.isOpened():
        print("Frame_id:", frame_id)

        # Get next frame
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

        # Inside the frame processing loop:
        for idx,track in enumerate(tracks):
            xmin, ymin, xmax, ymax, track_id = track
            cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2  # Centroid
            tracking_data.append([frame_id, track_id, labels[idx], cx, cy, xmin, ymin, xmax, ymax]) 

        # Display the frame
        #cv2.imshow('YOLOv5 Inference', frame)

        # Wait for a key press (25ms delay between frames)
        # Press 'q' to exit the sequence
        if cv2.waitKey(25) & 0xFF == ord('q') or frame_id == 1500:
            break

        frame_id += 1

    # Release the video capture object and close windows
    cap.release()
    cv2.destroyAllWindows()

    # Save tracking data to a CSV file
    df = pd.DataFrame(tracking_data, columns=['frame_id', 'track_id', 'class', 'cx', 'cy', 'xmin', 'ymin', 'xmax', 'ymax'])
    df.to_csv('../results/video_predicted/tracking/tracking_' + name_video + '.csv', index=False)
    
    
    
def calculate_centroid_velocity(name_video):
    # Load the tracking data from a CSV file
    df = pd.read_csv('../results/video_predicted/tracking/tracking_' + name_video + '.csv')

    # Calculate velocity for each track_id
    df['velocity_x'] = 0.0
    df['velocity_y'] = 0.0
    df['speed'] = 0.0

    # Frame rate of the video (frames per second)
    dt = 1 / fps  # Time interval between frames

    # Group by track_id and calculate velocity
    for track_id, group in df.groupby('track_id'):
        # Calculate displacement (delta x and delta y)
        group['delta_x'] = group['cx'].diff()
        group['delta_y'] = group['cy'].diff()

        # Calculate velocity (pixels per second)
        group['velocity_x'] = np.round(group['delta_x'] / dt,2)
        group['velocity_y'] = np.round(group['delta_y'] / dt,2)

        # Calculate speed (magnitude of velocity)
        group['speed'] = np.round((group['velocity_x']**2 + group['velocity_y']**2)**0.5,2)
        
        
        # Calculate mean and maximum velocity
        group["mean_velocity"] = np.round(group['speed'].mean(),2)
        group["max_velocity"] = np.round(group['speed'].max(),2)
        
        df.loc[group.index, ['mean_velocity', 'max_velocity']] = group[['mean_velocity', 'max_velocity']].fillna(0)
        
        # Update the original DataFrame
        df.loc[group.index, ['velocity_x', 'velocity_y', 'speed']] = group[['velocity_x', 'velocity_y', 'speed']].fillna(0)

    # Save the updated DataFrame with velocity data
    df.to_csv('../results/video_predicted/centroid_velocity/centroid_velocity_' + name_video + '.csv', index=False)

    print("Velocity data saved to sperm_tracking_with_velocity")
    
def show_video_tracking(video_path,name_video):
    # Load the tracking data with velocity
    df = pd.read_csv('../results/video_predicted/centroid_velocity/centroid_velocity_' + name_video + '.csv')
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    trajectories = {}

    # Set the slow-motion factor (e.g., 0.5 for half speed, 0.25 for quarter speed)
    slow_motion_factor = 0.3 # Adjust this value as needed

    # Calculate the new delay between frames
    original_delay = int(1000 / fps)  # Delay in milliseconds
    new_delay = int(original_delay / slow_motion_factor)

    # Process the video frame by frame
    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Get the data for the current frame
        frame_data = df[df['frame_id'] == frame_id]

        # Draw velocity vectors on the frame
        for _, row in frame_data.iterrows():
            cx, cy = int(row['cx']), int(row['cy'])
            vx, vy = row['velocity_x'], row['velocity_y']

            # Scale the velocity vector for visualization
            scale = 0.3  # Adjust this to make the vectors visible
            end_point = int(cx + vx * scale), int(cy + vy * scale)
            track_id = row['track_id']
            if track_id not in trajectories:
                trajectories[track_id] = []
            trajectories[track_id].append((cx, cy))
        
            # Draw path
            for i in range(1, len(trajectories[track_id])):
                cv2.line(frame, (int(trajectories[track_id][i - 1][0]),int(trajectories[track_id][i - 1][1])), (int(trajectories[track_id][i][0]),int(trajectories[track_id][i][1])), (0, 255, 0), 1)
                
            cv2.putText(frame, str(int(track_id)), (cx + 10, cy), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 255), 1,  cv2.LINE_AA )
            cv2.rectangle(frame, (int(row['xmin']), int(row['ymin'])), (int(row['xmax']), int(row['ymax'])), (0,255,0))
            
            # Draw the velocity vector
            #cv2.arrowedLine(frame, (cx, cy), end_point, (0, 255, 0), 2)
            #cv2.putText(frame, str(round(row['speed']*pixel_to_micron,2)) + "µm/s", (cx + 10, cy), cv2.FONT_HERSHEY_PLAIN, 0.6, (0, 255, 0), 2)


        # Display the frame
        cv2.imshow('Sperm Velocity', frame)

        # Save the frame (optional)
        cv2.imwrite(f'output/frame_{frame_id:04d}.jpg', frame)

        # Wait for the calculated delay
        if cv2.waitKey(new_delay) & 0xFF == ord('q'):
            break
        
        frame_id += 1

    # Release the video capture object and close windows
    cap.release()
    cv2.destroyAllWindows()


def calculate_features(name_video):
    # Load the tracking data from a CSV file
    df = pd.read_csv('../results/video_predicted/centroid_velocity/centroid_velocity_' + name_video + '.csv')

    columns = ['sperm_id','total_distance','displacement','time_elapsed','vcl','vsl','vap','alh','mad','lin','wob','str','bcf']
    data = pd.DataFrame(columns=columns)

    # Group by track_id and calculate velocity
    for track_id, group in df.groupby('track_id'):
        if len(group) >= 25:
            # Convert the columns to a list of tuples
            trajectory_path = list(zip(group['cx'], group['cy']))
            
            # Basic measures
            time_elapsed = calculate_time_elapsed(trajectory_path,fps)
            displacement = calculate_displacement(trajectory_path)
            total_distance = calculate_total_distance(trajectory_path)

            # Standard measures
            vcl = calculate_VCL(trajectory_path,fps)
            vsl = calculate_VSL(trajectory_path,fps)
            vap = calculate_VAP(trajectory_path,fps)
            alh = calculate_ALH(trajectory_path)
            mad = calculate_MAD(trajectory_path)
            
            # Commonly measures
            linearity = calculate_linearity(trajectory_path,fps)
            wob = calculate_WOB(trajectory_path,fps)
            straightness = calculate_STR(trajectory_path,fps)
            bcf = calculate_BCF(trajectory_path,fps)
            #curvature = calculate_curvature(trajectory_path)

            new_row = pd.DataFrame([[track_id,total_distance,displacement,time_elapsed,vcl,vsl,vap,alh,mad,linearity,wob,straightness,bcf]], columns=data.columns)
            data = pd.concat([data,new_row], ignore_index=True)

    # Save the DataFrame
    data.to_csv('../results/video_predicted/features/features_' + name_video + '.csv', index=False)


def preprocessing_data(name_video):
    
    # Load the tracking data from a CSV file
    df = pd.read_csv('../results/video_predicted/features/features_' + name_video + '.csv')
    
    df = df.drop('sperm_id', axis=1)
    df_cleaned = deleted_null_values(df)
    df_scaler = scaler(df_cleaned)
    #df_cleaned_outliers = iqr_median_impute(df_scaler, exclude_cols=['label'])
    
    df = pd.DataFrame(df_scaler, columns=['total_distance','displacement','time_elapsed','vcl','vsl','vap','alh','mad','lin','wob','str','bcf'])
    
    # Save the updated DataFrame with velocity data
    df.to_csv('../results/video_predicted/preprocessing/' + name_video + '_preprocessing.csv', index=False)
    

def classify_data(name_video):
    # Load model
    loaded_model = joblib.load('../models/simple_NN_2c.joblib')
    
    # Load data
    df = pd.read_csv('../results/video_predicted/preprocessing/' + name_video + '_preprocessing.csv')
    df2 = pd.read_csv('../results/video_predicted/features/features_' + name_video + '.csv')
    
    # Delete unused column
    X = df[['vcl', 'vsl', 'vap', 'alh', 'str']]
    
    print(X)
    # Mapping of numeric values to class names
    class_names = {
        0: "Progressive",
        1: "Non-progressive"
    }
    
    # Predict
    y_pred = np.array((loaded_model.predict(X) > 0.5).astype("int32")).flatten().tolist()
    #y_pred=loaded_model.predict(X)
    
    # Replace numeric values with class names
    print(y_pred)
    y_pred_mapped = [class_names[label] for label in y_pred]
    
    # Create a count plot with different colors per class
    ax = sns.countplot(x=y_pred_mapped, palette="Set2")

    # Add the count labels on top of each bar
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', 
                    fontsize=12, color='black', 
                    xytext=(0, 5), textcoords='offset points')

    plt.title("Distribution of Sperm Motility Categories")
    plt.xlabel("Categories")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()
    
    
    df2['label'] = y_pred
    
    df2.to_csv("aa_2.csv")
    
  
def show_video_tracking_labels(video_path,name_video):
    # Load the tracking data with velocity
    df = pd.read_csv('../results/video_predicted/centroid_velocity/centroid_velocity_' + name_video + '.csv')
    # Load the tracking data with velocity
    df_class = pd.read_csv('./aa_2.csv')
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    trajectories = {}

    # Set the slow-motion factor (e.g., 0.5 for half speed, 0.25 for quarter speed)
    slow_motion_factor = 0.3 # Adjust this value as needed

    # Calculate the new delay between frames
    original_delay = int(1000 / fps)  # Delay in milliseconds
    new_delay = int(original_delay / slow_motion_factor)

    # Process the video frame by frame
    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Get the data for the current frame
        frame_data = df[df['frame_id'] == frame_id]
        total_class = 3

        # Draw velocity vectors on the frame
        for _, row in frame_data.iterrows():
            cx, cy = int(row['cx']), int(row['cy'])

            # Scale the velocity vector for visualization
            track_id = row['track_id']
            if track_id not in trajectories:
                trajectories[track_id] = []
            trajectories[track_id].append((cx, cy))
            
            # Set class
            n_class = 0
            try:
                n_class = int(df_class[df_class['sperm_id']==track_id]['label'])
            except:
                continue
            
            # Define the color according to the class. EstruQcture: 2c/3c/4c
            color = (0, 0, 0)
            if n_class == 0:
                color = (0, 255, 0) # Green - Progressive
            elif n_class == 1:
                color = (255, 0, 0) # Blue - Non progressive
        
            # Draw path
            for i in range(1, len(trajectories[track_id])):
                cv2.line(frame, (int(trajectories[track_id][i - 1][0]),int(trajectories[track_id][i - 1][1])), (int(trajectories[track_id][i][0]),int(trajectories[track_id][i][1])), color, 1)
                
            cv2.putText(frame, str(int(track_id)), (cx + 10, cy), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 255), 1,  cv2.LINE_AA )
            cv2.rectangle(frame, (int(row['xmin']), int(row['ymin'])), (int(row['xmax']), int(row['ymax'])), color)
            
            # Draw the velocity vector
            #cv2.arrowedLine(frame, (cx, cy), end_point, (0, 255, 0), 2)
            #cv2.putText(frame, str(round(row['speed']*pixel_to_micron,2)) + "µm/s", (cx + 10, cy), cv2.FONT_HERSHEY_PLAIN, 0.6, (0, 255, 0), 2)


        # Display the frame
        cv2.imshow('Sperm Velocity', frame)

        # Wait for the calculated delay
        if cv2.waitKey(new_delay) & 0xFF == ord('q'):
            break
        
        frame_id += 1

    # Release the video capture object and close windows
    cap.release()
    cv2.destroyAllWindows()
        
    
def classify_video(video_path,name_video):
    
    #traking_video(video_path,name_video)
    #calculate_centroid_velocity(name_video)
    #show_video_tracking(video_path,name_video)
    #calculate_features(name_video)
    #preprocessing_data(name_video)
    classify_data(name_video)
    show_video_tracking_labels(video_path,name_video)
    

if __name__ == "__main__":
    # Path to video
    video_path = '../data/VISEM_Tracking/train/11/11.mp4'
    
    classify_video(video_path,'Prueba_clasificacion_11')
    
    