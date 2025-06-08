import seaborn as sns
from sort.sort import *
import pandas as pd
import matplotlib.pyplot as plt
from classify_by_movement import *
import pandas as pd
from functions_features import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer

def scaler(df):
    scaler = MinMaxScaler()
    columns=['total_distance','displacement','time_elapsed','vcl','vsl','vap','alh','mad','lin','wob','str','bcf']
    df[columns] = np.round(scaler.fit_transform(df[columns].select_dtypes(include=['float64', 'int64'])), 4)
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
    columns=['total_distance','displacement','time_elapsed','vcl','vsl','vap','alh','mad','lin','wob','str','bcf']
    
    iso = IsolationForest(contamination=contamination, random_state=42)
    preds = iso.fit_predict(df[columns])

    # Keep only inliers (label 1)
    df_clean = df[preds == 1].reset_index(drop=True)
    
    print(df.shape)
    print(df_clean.shape)
    
    return df_clean



def iqr_median_impute(df, exclude_cols=None, max_iter=10):
    if exclude_cols is None:
        exclude_cols = []

    df_clean = df.copy()
    numeric_cols = [col for col in df_clean.select_dtypes(include=[np.number]).columns if col not in exclude_cols]

    for col in numeric_cols:
        for _ in range(max_iter):
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            outliers = (df_clean[col] < lower) | (df_clean[col] > upper)
            if not outliers.any():
                break

            median = df_clean.loc[~outliers, col].median()
            df_clean.loc[outliers, col] = median

    return df_clean

def preprocessing_dataset(df,name_file_result):
    df = df.drop('sperm_id', axis=1)
    df_cleaned = deleted_null_values(df)
    df_scaler = scaler(df_cleaned)
    df_cleaned_outliers = iqr_median_impute(df_scaler, exclude_cols=['label'])
    
    df = pd.DataFrame(df_cleaned_outliers, columns=['total_distance','displacement','time_elapsed','vcl','vsl','vap','alh','mad','lin','wob','str','bcf','label'])
    
    # Save the updated DataFrame with velocity data
    df.to_csv('../results/data_features_labelling_preprocessing/' + name_file_result + '.csv', index=False)
    
    return df
    
    
if __name__ == "__main__":
    # Load the tracking data from a CSV file
    df = pd.read_csv('../results/data_features_labelling/dataset_3c_30s_15_3_01.csv')
    
    df = df.drop('sperm_id', axis=1)
    df_cleaned = deleted_null_values(df)
    df_scaler = scaler(df_cleaned)
    df_cleaned_outliers = iqr_median_impute(df_scaler, exclude_cols=['label'])
    
    df = pd.DataFrame(df_cleaned_outliers, columns=['total_distance','displacement','time_elapsed','vcl','vsl','vap','alh','mad','lin','wob','str','bcf','label'])
    
    # Save the updated DataFrame with velocity data
    df.to_csv('../results/data_features_labelling_preprocessing/dataset_3c_30s_15_3_01_preprocessing.csv', index=False)