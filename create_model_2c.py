

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

import warnings
warnings.filterwarnings("ignore")

def draw_confusion_matrix(y_test,y_pred):
    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    # Create a heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-progressive', 'Progressive'], 
                yticklabels=['Non-progressive', 'Progressive'])

    # Add labels and title
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix')

    # Show the plot
    plt.show()
    
    
def draw_roc_auc_curve(model,X_test,y_test):
    # Probabilidades de clase positiva
    y_prob = model.predict_proba(X_test)[:, 1]

    # Calcular curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    # Graficar
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # Línea base
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

def show_metrics(y_test,y_pred):
    # Métricas
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    

def random_forest(df):
    # Features and labels
    X = df.drop(["Label","sperm_id"], axis=1).values
    y = df["Label"]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest classifier
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    
    # Show metrics
    show_metrics(y_test,y_pred)
    draw_confusion_matrix(y_test,y_pred)
    draw_roc_auc_curve(model,X_test,y_test)
    
    dump(model, "models/random_forest.joblib")
    

if __name__ == "__main__":
    # Load the tracking data from a CSV file
    df = pd.read_csv('results/data_features_labeling/dataset.csv')
    
    random_forest(df)