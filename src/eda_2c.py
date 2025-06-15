import seaborn as sns
from sort.sort import *
import pandas as pd
import matplotlib.pyplot as plt
from classify_by_movement import *
import pandas as pd
from functions_features import *

matplotlib.use("TkAgg")  # Use Tkinter-based backend

    
def draw_class_distribution(df):
    # Count the number of samples for each class
    label_counts = df["label"].value_counts()

    # Plot the label distribution
    plt.figure(figsize=(8, 6))
    sns.barplot(x=label_counts.index, y=label_counts.values, hue=label_counts.index, palette="viridis", legend=False)
    plt.title("Label Distribution (Progressive vs. Non-Progressive)", fontsize=16)
    plt.xlabel("Label (0 = Progressive, 1 = Non-Progressive)", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    plt.xticks([0, 1], ["Progressive", "Non-Progressive"])  # Replace 0 and 1 with meaningful labels
    plt.show()
        
def draw_correlation_matix(df):
    # Calculate the correlation matrix
    corr_matrix = df.corr()

    # Plot the correlation matrix as a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Matrix", fontsize=16)
    plt.show()


def draw_distribucion_columns(df):
    # Distribution of numerical features
    df.hist(bins=20, figsize=(12, 10))
    plt.show()
    
def show_outliers(df):
   # Select only numeric columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

    # Calculate the number of rows and columns for the subplot grid
    num_cols = 3  # You can set this to a number you prefer
    num_rows = int(np.ceil(len(numeric_cols) / num_cols))

    # Set up the matplotlib figure
    plt.figure(figsize=(num_cols * 5, num_rows * 5))

    # Iterate over each numeric column to plot a boxplot
    for idx, column in enumerate(numeric_cols):
        plt.subplot(num_rows, num_cols, idx + 1)  # Dynamically position in grid
        sns.boxplot(data=df, x=column)
        plt.title(f'Boxplot for {column}')

    # Adjust space between subplots
    plt.subplots_adjust(hspace=1, wspace=1)  # Increase these values for more space

    plt.tight_layout()
    plt.show()
    

if __name__ == "__main__":
    # Load the tracking data from a CSV file
    df = pd.read_csv('../results/data_features_labelling_preprocessing/dataset_30s_2c.csv')
    
    draw_class_distribution(df)
    #draw_correlation_matix(df)
    #draw_distribucion_columns(df)
    #show_outliers(df)