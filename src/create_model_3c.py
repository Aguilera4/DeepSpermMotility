import seaborn as sns
from sort.sort import *
import pandas as pd
import matplotlib.pyplot as plt
from classify_by_movement import *
import pandas as pd
from functions_features import *
from sklearn.model_selection import GridSearchCV
from joblib import dump
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers
from sklearn.model_selection import learning_curve
import numpy as np
import os
import warnings
from sklearn.preprocessing import label_binarize
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from imblearn.under_sampling import RandomUnderSampler
from tabpfn import TabPFNClassifier
from tabpfn_extensions.rf_pfn import (
    RandomForestTabPFNClassifier,
    RandomForestTabPFNRegressor,
)
import joblib
from sklearn.decomposition import PCA
from sklearn.utils import class_weight
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel,RFE
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay



matplotlib.use("TkAgg")  # Use Tkinter-based backend
warnings.filterwarnings("ignore")




def draw_confusion_matrix(y_test,y_pred):
    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    # Create a heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Progressive motility', 'Non-pogressive', 'Inmotile'], 
                yticklabels=['Progressive motility', 'Non-pogressive', 'Inmotile'])

    # Add labels and title
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix')

    # Show the plot
    plt.show()
    

def show_metrics(y_test,y_pred):
    # Evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')  # or 'micro', 'weighted'
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    
def show_learning_curve(results):
    # Plot learning curves
    epochs = len(results['validation_0']['mlogloss'])
    x_axis = range(epochs)
        
    # Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, results['validation_0']['auc'], label='Train')
    plt.plot(x_axis, results['validation_1']['auc'], label='Test')
    plt.xlabel('Epochs')
    plt.ylabel('auc')
    plt.title('XGBoost Learning Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

    # mlogloss
    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, results['validation_0']['mlogloss'], label='Train')
    plt.plot(x_axis, results['validation_1']['mlogloss'], label='Test')
    plt.xlabel('Epochs')
    plt.ylabel('Log Loss')
    plt.title('XGBoost Learning Curve')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    '''# merror
    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, results['validation_0']['merror'], label='Train')
    plt.plot(x_axis, results['validation_1']['merror'], label='Test')
    plt.xlabel('Epochs')
    plt.ylabel('merror')
    plt.title('XGBoost Learning Curve')
    plt.legend()
    plt.grid(True)
    plt.show()'''

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
    

def plot_learning_curve_RF(clf,X,y):
    
    train_sizes, train_scores, test_scores = learning_curve(clf, X, y, cv=5, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 10), shuffle=True, random_state=42)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std  = np.std(train_scores, axis=1)
    test_scores_mean  = np.mean(test_scores, axis=1)
    test_scores_std   = np.std(test_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.title("Curva de aprendizaje - Random Forest")
    plt.xlabel("Tamaño del conjunto de entrenamiento")
    plt.ylabel("Exactitud (Accuracy)")

    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Entrenamiento")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Validación")

    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()
    
def select_features_importance(X_train,y_train,X_test):
    # Use SelectFromModel to select features based on importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rfe = RFE(rf, n_features_to_select=5)
    rfe.fit(X_train, y_train)
    
    # Importances
    selected_features = rfe.get_support()
    selected_names = X_train.columns[selected_features]

    print("Selected features:", list(selected_names))
    
    X_train_selected = rfe.transform(X_train)
    X_test_selected = rfe.transform(X_test)

    return [X_train_selected,X_test_selected]


def draw_roc_auc_curve(y_test,y_pred_prob):
    # Compute ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = 3
    
    y_test = np.array(y_test)
    
    # Reshape if 1D
    if y_test.ndim == 1:
        y_test = y_test.reshape(-1, 1)
        
    y_test = label_binarize(y_test, classes=[0, 1, 2, 3])

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot all ROC curves
    plt.figure(figsize=(8, 6))
    colors = ['blue', 'green', 'red']
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], color=colors[i],
                label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')  # diagonal line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multiclass ROC Curve (OvR)')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()

############################## MODELS ##############################

def random_forest(df):
    # Features and labels
    X = df.drop(["label"], axis=1)
    y = LabelEncoder().fit_transform(df['label'])

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Apply SMOTE to balance
    smote = SMOTE(k_neighbors=2, random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    # Features importance
    X_train_selected, X_test_selected = select_features_importance(X_train_resampled,y_train_resampled,X_test)
    
    # Train a Random Forest classifier
    clf = RandomForestClassifier(n_estimators=100, max_depth=40, random_state=42)
    clf.fit(X_train_selected, y_train_resampled)

    # Evaluate the model
    y_pred = clf.predict(X_test_selected)
    
    # Show metrics
    show_metrics(y_test,y_pred)
    draw_confusion_matrix(y_test,y_pred)
    
    #dump(clf, "../models/random_forest_4c.joblib")
    
def logistic_regression(df):
    # Features and labels
    X = df.drop(["label"], axis=1)
    y = df["label"]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    
    # Apply SMOTE to balance
    smote = SMOTE(k_neighbors=2, random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    # Features importance
    X_train_selected, X_test_selected = select_features_importance(X_train_resampled,y_train_resampled,X_test)
    
    # Train a Logistic regression model
    clf = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=100)
    clf.fit(X_train_selected, y_train_resampled)

    # Evaluate the model
    y_pred = clf.predict(X_test_selected)
    y_pred_prob = clf.predict_proba(X_test_selected) # Probability estimates for the positive class
    
    # Show metrics
    show_metrics(y_test,y_pred)
    draw_confusion_matrix(y_test,y_pred)
    draw_roc_auc_curve(y_test,y_pred_prob)
    
    #dump(clf, "../models/linear_regression_4c.joblib")
    

def XGBoost(df):
    # Features and labels
    X = df.drop(["label"], axis=1)
    y = LabelEncoder().fit_transform(df['label'])
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    
    # Apply SMOTE to balance
    smote = SMOTE(k_neighbors=2, random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    # Features importance
    X_train_selected, X_test_selected = select_features_importance(X_train_resampled,y_train_resampled,X_test)
    
    # Define parameter grid for GridSearchCV
    param_grid = {
        'n_estimators': [50, 100, 200, 1000],
        'learning_rate': [0.01, 0.1, 0.3],
        'max_depth': [3, 5, 7]
    }
    
    #X_train_selected, X_test_selected = select_features_importance(X_train_resampled,y_train_resampled,X_test,X.columns)
    
    # Initialize XGBoost classifier
    model = xgb.XGBClassifier(
        objective="multi:softmax",
        num_class=3,
        eval_metric=["mlogloss", "auc"],
        learning_rate=0.1,
        max_depth=5,
        n_estimators=1000,
        random_state=42
    )
    
    '''
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=5, verbose=1, n_jobs=-1)
    '''
    
    # eval_set
    eval_set = [(X_train_selected, y_train_resampled), (X_test_selected, y_test)]
    
    # Train the model
    model.fit(X_train_selected, y_train_resampled, eval_set=eval_set, verbose=False)
    
    '''grid_search.fit(X_train, y_train)
    
    # Print the best parameters and best score
    print("Best Parameters: ", grid_search.best_params_)
    print("Best Cross-validation Accuracy: ", grid_search.best_score_)

    # Test the best model
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test_selected)

    # Evaluate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy on Test Set: {accuracy:.4f}')'''
    
    '''importances = model.feature_importances_
    #feature_names = df.drop(["label","sperm_id","displacement","time_elapsed","mad","wob","str","bcf","alh","total_distance","lin"], axis=1).columns  # Or provide a list if it's a NumPy array
    feature_names = df.drop(["label"], axis=1).columns  # Or provide a list if it's a NumPy array

    # Sort by importance
    indices = np.argsort(importances)[::-1]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), ['PCA_1','PCA_2'], rotation=45, ha='right')
    plt.tight_layout()
    plt.show()'''

    # Predict
    y_pred = model.predict(X_test_selected)

    # Show metrics
    show_metrics(y_test,y_pred)
    draw_confusion_matrix(y_test,y_pred)
    show_learning_curve(model.evals_result())
    
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print("Cross-validation scores:", cv_scores)
    print("Mean accuracy:", cv_scores.mean())
    
    #dump(model, "../models/XGBoost_4c_extended.joblib")


def tabPFN(df):
    label_encoder =  LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['label'])
    
    # Features and labels
    #X = df.drop(["label","displacement","time_elapsed","mad","wob","str","bcf"], axis=1).values.astype(np.float32)
    X = df.drop(["label"], axis=1)
    y = df["label"]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Apply SMOTE to balance
    smote = SMOTE(k_neighbors=2, random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        
    # Initialize a classifier
    clf_base =  TabPFNClassifier(
        ignore_pretraining_limits=True,
        inference_config = {"SUBSAMPLE_SAMPLES": 1000} # Needs to be set low so that not OOM on fitting intermediate nodes
    )
    
    tabpfn_tree_clf = RandomForestTabPFNClassifier(
        tabpfn=clf_base,
        verbose=1,
        max_predict_time=60, # Will fit for one minute
        fit_nodes=True, # Wheather or not to fit intermediate nodes
        adaptive_tree=True, # Whather or not to validate if adding a leaf helps or not
    )

    # Train the model
    tabpfn_tree_clf.fit(X_train_resampled, y_train_resampled)
    
    # Predict probabilities
    prediction_probabilities = tabpfn_tree_clf.predict_proba(X_test)
    print("ROC AUC:", roc_auc_score(y_test, prediction_probabilities[:, 1]))

    # Predict labels
    predictions = tabpfn_tree_clf.predict(X_test)
    print("Accuracy", accuracy_score(y_test, predictions))


    dump(tabpfn_tree_clf, "../models/TabPFN_3c_15s_extended.joblib")
    
    
def tabPFN_load():
    
    loaded_model = joblib.load('../models/TabPFN_4c_15s_extended.joblib')
    
    X_train = pd.read_csv('../results/train_test_split/X_train.csv')
    X_test = pd.read_csv('../results/train_test_split/X_test.csv')
    y_train = pd.read_csv('../results/train_test_split/y_train.csv')
    y_test = pd.read_csv('../results/train_test_split/y_test.csv')
    
    print(type(X_test))
    
    # Predict probabilities
    prediction_probabilities = loaded_model.predict_proba(X_test.iloc[[0]])
    print("ROC AUC:", roc_auc_score(y_test.iloc[[0]], prediction_probabilities[:, 1]))

    # Predict labels
    predictions = loaded_model.predict(X_test.iloc[[0]])
    print("Accuracy", accuracy_score(y_test.iloc[[0]], predictions))
    
    

def simple_NN(df):
    X = df.drop(columns=['label'])
    y = keras.utils.to_categorical(df['label'].values, 3)  # One-hot encoding for 3 classes

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # SMOTE
    smote = SMOTE(k_neighbors=2,random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    # Features importance
    X_train_selected, X_test_selected = select_features_importance(X_train_resampled,y_train_resampled,X_test)
    
    model = keras.Sequential([
        layers.InputLayer(input_shape=(X_train_selected.shape[1],)),  # Input layer for the features
        layers.Dense(128, activation='relu'),  # First hidden layer with 128 neurons
        layers.Dropout(0.3),  # Dropout for regularization to prevent overfitting
        layers.Dense(64, activation='relu'),  # Second hidden layer with 64 neurons
        layers.Dropout(0.3),  # Dropout for regularization
        layers.Dense(32, activation='relu'),  # Third hidden layer with 32 neurons
        layers.Dense(3, activation='softmax')  # Output layer with 3 classes and softmax activation
    ])
    
    
    # Compile model
    model.compile(optimizer=optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train the model normally
    history = model.fit(X_train_selected, y_train_resampled, epochs=100, batch_size=32, validation_data=(X_test_selected, y_test), callbacks=[early_stopping])
    
    '''
    model.layers[0].trainable = True      
    model.layers[1].trainable = True  

    # Compile the model with a lower learning rate for fine-tuning
    model.compile(optimizer=optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    # Continue training with fine-tuning for 5 more epochs
    history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test_selected, y_test))
    '''

    # Predict
    loss, accuracy = model.evaluate(X_test_selected, y_test)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    
    plot_learning_curve_NN(history)
    
    y_pred = (model.predict(X_test_selected) > 0.5).astype("int32")
    
    print(classification_report(y_test, y_pred))
    
    # Show metrics
    show_metrics(y_test,y_pred)
    draw_confusion_matrix(y_test,y_pred)
    
    #dump(model, "../models/simple_NN_4c_extended.joblib")



if __name__ == "__main__":
    # Load the tracking data from a CSV file
    df = pd.read_csv('../results/data_features_labelling_preprocessing/dataset_3c_30s_preprocessing.csv')
    
    #random_forest(df)
    #logistic_regression(df)
    #XGBoost(df)
    simple_NN(df)
    #tabPFN(df)
    #tabPFN_load()