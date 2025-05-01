import seaborn as sns
from sort.sort import *
import pandas as pd
import matplotlib.pyplot as plt
from classify_by_movement import *
import pandas as pd
from calculate_features import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest

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


def remove_outliers_isolation_forest(df, contamination=0.05):
    columns=['total_distance','displacement','time_elapsed','vcl','vsl','vap','alh','mad','linearity','wob','straightness','bcf','angular_displacement','curvature']
    
    iso = IsolationForest(contamination=contamination, random_state=42)
    preds = iso.fit_predict(df[columns])

    # Keep only inliers (label 1)
    df_clean = df[preds == 1].reset_index(drop=True)
    
    print(df.shape)
    print(df_clean.shape)
    
    return df_clean
       
    
if __name__ == "__main__":
    # Load the tracking data from a CSV file
    df = pd.read_csv('../results/data_features_labelling/dataset_extended_4c_30s.csv')
    
    df = df.drop('sperm_id', axis=1)
    
    df_cleaned = deleted_null_values(df)
    df_scaler = scaler(df_cleaned)
    df_cleaned_outliers = remove_outliers_isolation_forest(df_scaler)
    
    df = pd.DataFrame(df_cleaned_outliers, columns=['total_distance','displacement','time_elapsed','vcl','vsl','vap','alh','mad','linearity','wob','straightness','bcf','angular_displacement','curvature','label'])
    
    # Save the updated DataFrame with velocity data
    df.to_csv('../results/data_features_labelling_preprocessing/dataset_extended_4c_30s_preprocessing_v2.csv', index=False)