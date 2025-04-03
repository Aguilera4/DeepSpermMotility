import seaborn as sns
from sort.sort import *
import pandas as pd
import matplotlib.pyplot as plt
from classify_by_movement import *
import pandas as pd
from calculate_features import *
from joblib import dump
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers
import numpy as np
import os
import warnings
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
matplotlib.use("TkAgg")  # Use Tkinter-based backend
warnings.filterwarnings("ignore")

def draw_confusion_matrix(y_test,y_pred):
    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    # Create a heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Linear mean swim', 'Circular swim', 'Hyperactivated', 'Inmotile'], 
                yticklabels=['Linear mean swim', 'Circular swim', 'Hyperactivated', 'Inmotile'])

    # Add labels and title
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix')

    # Show the plot
    plt.show()
    
    
def draw_roc_auc_curve(y_test,y_pred):
    # Compute ROC curve and ROC AUC for each class
    n_classes = 4
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], label=f"Class {i} (AUC = {roc_auc[i]:.2f})")

    # Plot random chance line
    plt.plot([0, 1], [0, 1], 'k--', label="Random Chance (AUC = 0.50)")

    # Configure plot
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multiclass ROC Curve")
    plt.legend(loc="lower right")
    plt.grid()
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train a Random Forest classifier
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    
    y_pred_proba = model.predict_proba(X_test)
    
    # Show metrics
    show_metrics(y_test,y_pred)
    draw_confusion_matrix(y_test,y_pred)
    #draw_roc_auc_curve(y_test,y_pred_proba)
    
    dump(model, "models/random_forest_4c.joblib")
    
    
    
def simple_NN(df):
    x_data = df.drop(columns=['Label']).values.astype(np.float32)
    y_data = keras.utils.to_categorical(df['Label'].values, 4)  # One-hot encoding for 4 classes
    
    #smote = SMOTE(sampling_strategy='auto', random_state=42)
    #x_resampled, y_resampled = smote.fit_resample(x_data, y_data)

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=42)
    
    model = keras.Sequential([
        layers.InputLayer(input_shape=(x_train.shape[1],)),  # Input layer for the features
        layers.Dense(128, activation='relu'),  # First hidden layer with 128 neurons
        layers.Dropout(0.2),  # Dropout for regularization to prevent overfitting
        layers.Dense(64, activation='relu'),  # Second hidden layer with 64 neurons
        layers.Dropout(0.2),  # Dropout for regularization
        layers.Dense(32, activation='relu'),  # Third hidden layer with 32 neurons
        layers.Dense(4, activation='softmax')  # Output layer with 4 classes and softmax activation
    ])
    
    # Compile model
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
  

    # Train the model normally
    model.fit(x_train, y_train, epochs=20, batch_size=32, validation_data=(x_test, y_test), callbacks=[early_stopping])
    
    '''
    model.layers[0].trainable = True  
    
    model.layers[1].trainable = True  

    # Compile the model with a lower learning rate for fine-tuning
    model.compile(optimizer=optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    # Continue training with fine-tuning for 5 more epochs
    model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))
    '''
    

if __name__ == "__main__":
    # Load the tracking data from a CSV file
    df = pd.read_csv('../results/data_features_labeling_preprocessing/dataset_4c_extended_preprocessing.csv')
    
    #random_forest(df)
    simple_NN(df)