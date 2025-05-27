import pandas as pd
from functions_features import *

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Video's frame rate
fps = 50  

def calculate_features(df,name_file):
    """
    Calculate the features associated to the sperm tracking.
    
    Args:
        df: dataframe with the sperm tracking
        name_file: name of the output file to save the results
    
    Returns:
        None
    """
     
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

            new_row = pd.DataFrame([[int(track_id),total_distance,displacement,time_elapsed,vcl,vsl,vap,alh,mad,linearity,wob,straightness,bcf]], columns=data.columns)
            data = pd.concat([data,new_row], ignore_index=True)

    # Save the DataFrame
    data.to_csv('../results/data_features/' + name_file + '.csv', index=False)
    
if __name__ == "__main__":
    # Load the tracking data from a CSV file
    df = pd.read_csv('../results/data_sperm_tracking/sperm_tracking_data_30s_v2.csv')
    
    calculate_features(df,'dataset_30s')