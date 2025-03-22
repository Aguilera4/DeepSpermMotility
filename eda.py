

import torch
import seaborn as sns
from ultralytics import YOLO
import cv2
from sort.sort import *
import pandas as pd
import matplotlib.pyplot as plt
import math
from torchvision.utils import draw_bounding_boxes
import torchvision
from classify_by_movement import *
import pandas as pd
from calculate_features import *
import os
from joblib import dump, load
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

matplotlib.use("TkAgg")  # Use Tkinter-based backend

    
def draw_class_distribution(df):
    # Count the number of samples for each class
    label_counts = df["Label"].value_counts()

    # Plot the label distribution
    plt.figure(figsize=(8, 6))
    sns.barplot(x=label_counts.index, y=label_counts.values, hue=label_counts.index, palette="viridis", legend=False)
    plt.title("Label Distribution (Progressive vs. Non-Progressive)", fontsize=16)
    plt.xlabel("Label (0 = Non-Progressive, 1 = Progressive)", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    plt.xticks([0, 1], ["Non-Progressive", "Progressive"])  # Replace 0 and 1 with meaningful labels
    plt.show()
        
def draw_correlation_matix(df):
    # Calculate the correlation matrix
    corr_matrix = df.corr()

    # Plot the correlation matrix as a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Matrix", fontsize=16)
    plt.show()

if __name__ == "__main__":
    # Load the tracking data from a CSV file
    df = pd.read_csv('results/data_features_labeling/dataset.csv')
    
    df = df.drop(columns=['sperm_id'])
    
    draw_class_distribution(df)
    draw_correlation_matix(df)