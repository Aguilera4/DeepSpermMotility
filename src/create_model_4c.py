import seaborn as sns
from sort.sort import *
import pandas as pd
import matplotlib.pyplot as plt
from classify_by_movement import *
import pandas as pd
from calculate_features import *
from joblib import dump
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers
import numpy as np
import os
import warnings
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from imblearn.under_sampling import RandomUnderSampler

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
    
    
def show_learning_curve(results):
    # Plot learning curves
    epochs = len(results['validation_0']['mlogloss'])
    x_axis = range(epochs)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, results['validation_0']['auc'], label='Train')
    plt.plot(x_axis, results['validation_1']['auc'], label='Test')
    plt.xlabel('Epochs')
    plt.ylabel('auc')
    plt.title('XGBoost Learning Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, results['validation_0']['mlogloss'], label='Train')
    plt.plot(x_axis, results['validation_1']['mlogloss'], label='Test')
    plt.xlabel('Epochs')
    plt.ylabel('Log Loss')
    plt.title('XGBoost Learning Curve')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, results['validation_0']['merror'], label='Train')
    plt.plot(x_axis, results['validation_1']['merror'], label='Test')
    plt.xlabel('Epochs')
    plt.ylabel('merror')
    plt.title('XGBoost Learning Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_learning_curve_NN(history):
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training vs Validation Accuracy')
    plt.show()
    
    
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training vs Validation Loss')
    plt.show()
    


############################## MODELS ##############################

def random_forest(df):
    # Features and labels
    X = df.drop(["label","sperm_id"], axis=1).values
    y = df["label"]

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
    
    dump(model, "../models/random_forest_4c.joblib")
    
    
    
    
def simple_NN(df):    
    X = df.drop(columns=['label','sperm_id']).values.astype(np.float32)
    y = keras.utils.to_categorical(df['label'].values, 4)  # One-hot encoding for 4 classes
    
    '''smote = SMOTE(sampling_strategy='auto', random_state=42)
    x_resampled, y_resampled = smote.fit_resample(X, y)'''

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = keras.Sequential([
        layers.InputLayer(input_shape=(X_train.shape[1],)),  # Input layer for the features
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
    history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])
    
    '''
    model.layers[0].trainable = True      
    model.layers[1].trainable = True  

    # Compile the model with a lower learning rate for fine-tuning
    model.compile(optimizer=optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    # Continue training with fine-tuning for 5 more epochs
    history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))
    '''

    # Predict
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    
    plot_learning_curve_NN(history)
    
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    print(classification_report(y_test, y_pred))
    
    dump(model, "../models/simple_NN_4c_extended.joblib")    
    

def XGBoost(df):
    label_encoder =  LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['label'])
    
    # Features and labels
    X = df.drop(["label","sperm_id"], axis=1).values.astype(np.float32)
    y = df["label"]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    
    eval_set = [(X_train, y_train), (X_test, y_test)]
    
    # Calculate the class weights
    class_weights = len(y_train) / (len(label_encoder.classes_) * y_train.value_counts())
    
       
    '''
    # Apply RandomUnderSampler to balance
    rus = RandomUnderSampler(random_state=42)
    X_train_resampled, y_train_resampled = rus.fit_resample(X_train, y_train)

    '''
    
    # Apply SMOTE to balance
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
        
    # Initialize XGBoost classifier
    model = xgb.XGBClassifier(
        scale_pos_weight=class_weights,
        objective="multi:softmax",
        num_class=label_encoder.classes_.shape[0],
        eval_metric=["mlogloss", "auc", "merror"],
        use_label_encoder=False,
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )

    # Train the model
    model.fit(X_train_resampled, y_train_resampled, eval_set=eval_set, verbose=False)

    # Predict
    y_pred = model.predict(X_test)

    # Show metrics
    show_metrics(y_test,y_pred)
    draw_confusion_matrix(y_test,y_pred)
    show_learning_curve(model.evals_result())
    
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print("Cross-validation scores:", cv_scores)
    print("Mean accuracy:", cv_scores.mean())
    
    dump(model, "../models/XGBoost_4c_extended.joblib")



if __name__ == "__main__":
    # Load the tracking data from a CSV file
    df = pd.read_csv('../results/data_features_labelling_preprocessing/dataset_4c_extended_preprocessing.csv')
    
    #random_forest(df)
    #XGBoost(df)
    #simple_NN(df)