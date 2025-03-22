import pandas as pd
from classify_by_movement import *
from calculate_features import *

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Video's frame rate
fps = 50  

def data_labelling_2c(df):
    # Frame rate of the video (frames per second)
    dt = 1 / fps  # Time interval between frames
        
    columns = ['sperm_id','Velocity','Straightness_Ratio','Angular_Displacement','Linearity','Curvature','ALH','BCF','Total_Distance','Displacement','Time_Elapsed','Label']
    data = pd.DataFrame(columns=columns)

    # Group by track_id and calculate velocity
    for track_id, group in df.groupby('track_id'):
        # Convert the columns to a list of tuples
        trajectory_path = list(zip(group['cx'], group['cy']))

        # Features
        veolicity_mean = calculate_velocity(trajectory_path,fps)
        straightness = calculate_straightness(trajectory_path)
        angular_displacement = calculate_angular_displacement(trajectory_path)
        linearity = calculate_linearity(trajectory_path)
        curvature = calculate_curvature(trajectory_path)
        alh = calculate_alh(trajectory_path)
        bcf = calculate_bcf(trajectory_path,fps)
        total_distance = calculate_total_distance(trajectory_path)
        displacement = calculate_displacement(trajectory_path)
        time_elapsed = calculate_time_elapsed(trajectory_path,fps)
        label = is_progressive(trajectory_path, fps, velocity_threshold=10, straightness_threshold=0.8)

        new_row = pd.DataFrame([[track_id,veolicity_mean,straightness,angular_displacement,linearity,curvature,alh,bcf,total_distance,displacement,time_elapsed,label]], columns=data.columns)
        data = pd.concat([data,new_row], ignore_index=True)

    # Save the DataFrame
    data.to_csv('results/data_features_labeling/dataset.csv', index=False)
    
    
def data_labelling_4c(df):
    # Frame rate of the video (frames per second)
    dt = 1 / fps  # Time interval between frames
        
    columns = ['sperm_id','Velocity','Straightness_Ratio','Angular_Displacement','Linearity','Curvature','ALH','BCF','Total_Distance','Displacement','Time_Elapsed','Label']
    data = pd.DataFrame(columns=columns)

    # Group by track_id and calculate velocity
    for track_id, group in df.groupby('track_id'):
        # Convert the columns to a list of tuples
        trajectory_path = list(zip(group['cx'], group['cy']))

        # Features
        veolicity_mean = calculate_velocity(trajectory_path,fps)
        straightness = calculate_straightness(trajectory_path)
        angular_displacement = calculate_angular_displacement(trajectory_path)
        linearity = calculate_linearity(trajectory_path)
        curvature = calculate_curvature(trajectory_path)
        alh = calculate_alh(trajectory_path)
        bcf = calculate_bcf(trajectory_path,fps)
        total_distance = calculate_total_distance(trajectory_path)
        displacement = calculate_displacement(trajectory_path)
        time_elapsed = calculate_time_elapsed(trajectory_path,fps)
        label = classification_4_classes(trajectory_path, fps, straightness_threshold=0.8)

        new_row = pd.DataFrame([[track_id,veolicity_mean,straightness,angular_displacement,linearity,curvature,alh,bcf,total_distance,displacement,time_elapsed,label]], columns=data.columns)
        data = pd.concat([data,new_row], ignore_index=True)

    # Save the DataFrame
    data.to_csv('results/data_features_labeling/dataset_4c.csv', index=False)
    
    
if __name__ == "__main__":
    # Load the tracking data from a CSV file
    df = pd.read_csv('results/data_sperm_tracking/sperm_tracking_data.csv')
    
    #data_labelling_2c(df)
    data_labelling_4c(df)