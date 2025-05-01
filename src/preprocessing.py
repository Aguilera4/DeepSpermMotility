import seaborn as sns
from sort.sort import *
import pandas as pd
import matplotlib.pyplot as plt
from classify_by_movement import *
import pandas as pd
from calculate_features import *
from sklearn.preprocessing import MinMaxScaler

def scaler(df):
    scaler = MinMaxScaler()
    columns=['total_distance','displacement','time_elapsed','vcl','vsl','vap','alh','mad','linearity','wob','straightness','bcf','angular_displacement','curvature']
    df[columns] = scaler.fit_transform(df[columns].select_dtypes(include=['float64', 'int64']))
    return df
    
    
def deleted_null_values(df):
    # Count the number of rows before dropping null values
    initial_row_count = len(df)

    # Drop rows with any null values
    df_cleaned = df.dropna()

    # Count the number of rows after dropping null values
    final_row_count = len(df_cleaned)

    # Calculate the number of rows dropped
    rows_dropped = initial_row_count - final_row_count

    print(f'Number of rows dropped: {rows_dropped}')
    
    return df_cleaned
    
    
    
if __name__ == "__main__":
    # Load the tracking data from a CSV file
    df = pd.read_csv('../results/data_features_labelling/dataset_extended_4c_15s.csv')
    
    df_cleaned = deleted_null_values(df)
    df_scaler = scaler(df_cleaned)
    
    df = pd.DataFrame(df_scaler, columns=['sperm_id','total_distance','displacement','time_elapsed','vcl','vsl','vap','alh','mad','linearity','wob','straightness','bcf','angular_displacement','curvature','label'])
    
    # Save the updated DataFrame with velocity data
    df.to_csv('../results/data_features_labelling_preprocessing/dataset_extended_4c_15s_preprocessing.csv', index=False)