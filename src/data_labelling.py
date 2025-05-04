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
        
    columns = ['sperm_id','total_distance','displacement','time_elapsed','vcl','vsl','vap','alh','mad','linearity','wob','straightness','bcf','angular_displacement','curvature','label']
    data = pd.DataFrame(columns=columns)

    # Group by track_id and calculate velocity
    for track_id, group in df.groupby('track_id'):
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
        angular_displacement = calculate_angular_displacement(trajectory_path)
        curvature = calculate_curvature(trajectory_path)
        
        # Label
        label = classification_2_classes(trajectory_path, fps)

        new_row = pd.DataFrame([[track_id,total_distance,displacement,time_elapsed,vcl,vsl,vap,alh,mad,linearity,wob,straightness,bcf,angular_displacement,curvature,label]], columns=data.columns)
        data = pd.concat([data,new_row], ignore_index=True)

    # Save the DataFrame
    data.to_csv('../results/data_features_labelling/dataset_2c_11.csv', index=False)
    
def data_labelling_3c(df):
    # Frame rate of the video (frames per second)
    dt = 1 / fps  # Time interval between frames
        
    columns = ['sperm_id','total_distance','displacement','time_elapsed','vcl','vsl','vap','alh','mad','linearity','wob','straightness','bcf','angular_displacement','curvature','label']
    data = pd.DataFrame(columns=columns)

    # Group by track_id and calculate velocity
    for track_id, group in df.groupby('track_id'):
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
        angular_displacement = calculate_angular_displacement(trajectory_path)
        curvature = calculate_curvature(trajectory_path)
        
        # Label
        label = classification_3_classes(trajectory_path, fps, straightness_threshold=0.8)

        new_row = pd.DataFrame([[track_id,total_distance,displacement,time_elapsed,vcl,vsl,vap,alh,mad,linearity,wob,straightness,bcf,angular_displacement,curvature,label]], columns=data.columns)
        data = pd.concat([data,new_row], ignore_index=True)

    # Save the DataFrame
    data.to_csv('../results/data_features_labelling/dataset_3c_extended.csv', index=False)
    
    
def data_labelling_4c(df):
    # Frame rate of the video (frames per second)
    dt = 1 / fps  # Time interval between frames
        
    columns = ['sperm_id','total_distance','displacement','time_elapsed','vcl','vsl','vap','alh','mad','linearity','wob','straightness','bcf','angular_displacement','curvature','label']
    data = pd.DataFrame(columns=columns)

    # Group by track_id and calculate velocity
    for track_id, group in df.groupby('track_id'):
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
        angular_displacement = calculate_angular_displacement(trajectory_path)
        curvature = calculate_curvature(trajectory_path)
        
        # Label
        label = classification_4_classes(trajectory_path, fps)

        new_row = pd.DataFrame([[track_id,total_distance,displacement,time_elapsed,vcl,vsl,vap,alh,mad,linearity,wob,straightness,bcf,angular_displacement,curvature,label]], columns=data.columns)
        data = pd.concat([data,new_row], ignore_index=True)

    # Save the DataFrame
    data.to_csv('../results/data_features_labelling/dataset_11_v3.csv', index=False)


def data_labelling_4c_v2(df):
    # Frame rate of the video (frames per second)
    dt = 1 / fps  # Time interval between frames
        
    columns = ['sperm_id','total_distance','displacement','time_elapsed','vcl','vsl','vap','alh','mad','linearity','wob','straightness','bcf','angular_displacement','curvature','label']
    data = pd.DataFrame(columns=columns)

    # Group by track_id and calculate velocity
    for track_id, group in df.groupby('track_id'):
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
        angular_displacement = calculate_angular_displacement(trajectory_path)
        curvature = calculate_curvature(trajectory_path)
        
        # Label
        label = classification_4_classes_v2(trajectory_path, fps)

        new_row = pd.DataFrame([[track_id,total_distance,displacement,time_elapsed,vcl,vsl,vap,alh,mad,linearity,wob,straightness,bcf,angular_displacement,curvature,label]], columns=data.columns)
        data = pd.concat([data,new_row], ignore_index=True)

    # Save the DataFrame
    data.to_csv('../results/data_features_labelling/dataset_4c_5s_v2.csv', index=False)
    
def data_labelling_4c_v3(df):
    # Frame rate of the video (frames per second)
    dt = 1 / fps  # Time interval between frames
        
    columns = ['sperm_id','total_distance','displacement','time_elapsed','vcl','vsl','vap','alh','mad','linearity','wob','straightness','bcf','angular_displacement','curvature','label']
    data = pd.DataFrame(columns=columns)

    # Group by track_id and calculate velocity
    for track_id, group in df.groupby('track_id'):
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
        angular_displacement = calculate_angular_displacement(trajectory_path)
        curvature = calculate_curvature(trajectory_path)
        
        # Label
        label = classification_4_classes_v4(trajectory_path, fps)

        new_row = pd.DataFrame([[track_id,total_distance,displacement,time_elapsed,vcl,vsl,vap,alh,mad,linearity,wob,straightness,bcf,angular_displacement,curvature,label]], columns=data.columns)
        data = pd.concat([data,new_row], ignore_index=True)

    # Save the DataFrame
    data.to_csv('../results/data_features_labelling/dataset_11_v3.csv', index=False)
    
    
if __name__ == "__main__":
    # Load the tracking data from a CSV file
    df = pd.read_csv('../results/data_sperm_tracking/sperm_tracking_data_5s_v2.csv')
    
    #data_labelling_2c(df)
    #data_labelling_4c(df)
    data_labelling_4c_v2(df)