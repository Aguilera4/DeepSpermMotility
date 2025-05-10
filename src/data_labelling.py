import pandas as pd
from classify_by_movement import *

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


def data_labelling(df,type_classification,name_file):
    """
    Label sperm according to calculated characteristics.
    
    Args:
        df: dataframe with the sperm features
        type_classification: type of classification applied
        name_file: name of the output file to save the results
    
    Returns:
        None
    """
    
    classification_classes = None
    if type_classification == '2c':
        classification_classes = classification_2_classes
    elif type_classification == '3c':
        classification_classes = classification_3_classes
    elif type_classification == '4c':
        classification_classes = classification_4_classes
    elif type_classification == '4c_v2':
        classification_classes = classification_4_classes_v2
    elif type_classification == '4c_v4':
        classification_classes = classification_4_classes_v4
        
    df['label'] = None
    for idx, row in df.iterrows():
        label = classification_classes(row)
        df['label'][idx] = label
        
    # Save the DataFrame
    df.to_csv('../results/data_features_labelling/' + name_file + '.csv', index=False)
    
    
if __name__ == "__main__":
    # Load the tracking data from a CSV file
    df = pd.read_csv('../results/data_features/dataset_35_v3.csv')
    
    #data_labelling(df,'2c','dataset_2c_11')
    data_labelling(df,'3c','dataset_3c_35_v5')
    #data_labelling(df,'2c','dataset_11_v3')
    #data_labelling(df,'2c','dataset_4c_5s_v2')
    #data_labelling(df,'4c_v2','dataset_4c_22_v2')