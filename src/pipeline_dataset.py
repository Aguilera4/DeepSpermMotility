import pandas as pd
from functions_features import *
import calculate_features
import data_labelling
import preprocessing

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
    
if __name__ == "__main__":
    
    
    tracking_file = 'sperm_tracking_data_11_30s_15_20_01.csv'
    
    print("Started pipeline...")
    
    # Load the tracking data from a CSV file
    df = pd.read_csv('../results/data_sperm_tracking/' + tracking_file)
    
    print("Started the calculation of characteristics ...")
    df_features = calculate_features.calculate_features(df,'dataset_11_30s_15_20_01')
    print("Completed the calculation of characteristics...")
    
    print("Started data labelling...")
    df_labelling = data_labelling.data_labelling(df_features,'3c','dataset_11_30s_15_20_01') # '2c','3c','4c','4c_v2','4c_v4'
    print("Completed data labelling...")
    
    print("Started preprocessing...")
    df_preprocessing = preprocessing.preprocessing_dataset(df_labelling,'dataset_11_30s_15_20_01')
    print("Completed preprocessing...")
    
    print("Completed pipeline...")