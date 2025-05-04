import seaborn as sns
from sort.sort import *
import pandas as pd
import matplotlib.pyplot as plt
from classify_by_movement import *
import pandas as pd
from calculate_features import *
from joblib import dump
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers
from sklearn.model_selection import learning_curve
import numpy as np
import os
import warnings
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
from sklearn.datasets import load_digits, load_wine
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer

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
    
    
def draw_roc_auc_curve(clf,X_test,y_test):
    # Obtener probabilidades
    y_score = clf.predict_proba(X_test)

    # Manejo de RandomForest multicapa de salida
    # Cada salida corresponde a una clase
    if isinstance(y_score, list):  # scikit-learn da una lista para multiclase
        y_score = np.stack([prob[:, 1] for prob in y_score], axis=1)

    # Curvas ROC y AUC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = 4

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Macro AUC (promedio de AUC por clase)
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    macro_auc = auc(all_fpr, mean_tpr)

    # Graficar
    plt.figure(figsize=(8, 6))
    colors = ['blue', 'green', 'orange']
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], color=colors[i],
                label=f'Clase {i} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel("Tasa de Falsos Positivos (FPR)")
    plt.ylabel("Tasa de Verdaderos Positivos (TPR)")
    plt.title(f"Curvas ROC por clase - AUC macro = {macro_auc:.2f}")
    plt.legend(loc="lower right")
    plt.grid()
    plt.tight_layout()
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
    
    
    # merror
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

############################## MODELS ##############################

def random_forest():
    # Step 2: Load the dataset
    # Download the dataset from UCI repository or use pandas to load if it's already local
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
    column_names = ['Age', 'WorkClass', 'Fnlwgt', 'Education', 'EducationNum', 'MaritalStatus', 'Occupation', 
                    'Relationship', 'Race', 'Sex', 'CapitalGain', 'CapitalLoss', 'HoursPerWeek', 'NativeCountry', 'Income']
    data = pd.read_csv(url, names=column_names, delimiter=',\s', engine='python')

    # Step 3: Handle missing data (if any) and encode categorical features
    # Checking for missing values
    data.isnull().sum()

    # Use SimpleImputer to fill missing values
    imputer = SimpleImputer(strategy='most_frequent')
    data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

    # Encode categorical columns
    label_encoder = LabelEncoder()

    # List of categorical columns
    categorical_cols = ['WorkClass', 'Education', 'MaritalStatus', 'Occupation', 'Relationship', 'Race', 'Sex', 'NativeCountry', 'Income']

    # Encode categorical columns into numerical values
    for col in categorical_cols:
        data_imputed[col] = label_encoder.fit_transform(data_imputed[col])

    # Step 4: Split the data into features (X) and target (y)
    X = data_imputed.drop('Income', axis=1)
    y = data_imputed['Income']

    # Split the dataset into training and test sets (80-20 split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 5: Standardize the features (optional)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Step 6: Train the model (Random Forest Classifier)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Step 7: Make predictions on the test set
    y_pred = model.predict(X_test)

    # Step 8: Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    
def logistic_regression(df):
    # Features and labels
    X = df.drop(["label"], axis=1).values
    y = df["label"]

    #pca = PCA(n_components=2)
    #X_pca = pca.fit_transform(X)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Apply SMOTE to balance
    smote = SMOTE(k_neighbors=2, random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    # Train a Logistic regression model
    clf = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
    clf.fit(X_train_resampled, y_train_resampled)

    # Evaluate the model
    y_pred = clf.predict(X_test)
    
    # Show metrics
    show_metrics(y_test,y_pred)
    draw_confusion_matrix(y_test,y_pred)
    
    dump(clf, "../models/linear_regression_4c.joblib")
    

def XGBoost():
    # Step 2: Load the dataset
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
    column_names = ['Age', 'WorkClass', 'Fnlwgt', 'Education', 'EducationNum', 'MaritalStatus', 'Occupation', 
                    'Relationship', 'Race', 'Sex', 'CapitalGain', 'CapitalLoss', 'HoursPerWeek', 'NativeCountry', 'Income']
    data = pd.read_csv(url, names=column_names, delimiter=',\s', engine='python')

    # Step 3: Handle missing data (if any) and encode categorical features
    data.isnull().sum()

    # Use SimpleImputer to fill missing values
    imputer = SimpleImputer(strategy='most_frequent')
    data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

    # Encode categorical columns
    label_encoder = LabelEncoder()
    categorical_cols = ['WorkClass', 'Education', 'MaritalStatus', 'Occupation', 'Relationship', 'Race', 'Sex', 'NativeCountry', 'Income']

    # Encode categorical columns into numerical values
    for col in categorical_cols:
        data_imputed[col] = label_encoder.fit_transform(data_imputed[col])

    # Step 4: Split the data into features (X) and target (y)
    X = data_imputed.drop('Income', axis=1)
    y = data_imputed['Income']
        
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    '''
    # Apply RandomUnderSampler to balance
    rus = RandomUnderSampler(random_state=42)
    X_train_resampled, y_train_resampled = rus.fit_resample(X_train, y_train)
    '''
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Initialize XGBoost classifier
    model = xgb.XGBClassifier(
        num_class=9,
        objective="multi:softmax",
        eval_metric=['mlogloss', 'auc'],
        use_label_encoder=False,
        n_estimators=1000,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )

    
    # eval_set
    evals = [(X_train, y_train), (X_test, y_test)]
    
    # Train the model
    history = model.fit(X_train, y_train, eval_set=evals, verbose=True)
    
    # Step 8: Make predictions on the test set
    y_pred = model.predict(X_test)

    # Step 9: Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Step 10: Plot training metrics (log-loss, accuracy, and error)
    # Get the log-loss, error, and accuracy history from training
    train_error = history.evals_result()['validation_0']['auc']
    test_error = history.evals_result()['validation_1']['auc']
    train_mlogloss = history.evals_result()['validation_0']['mlogloss']
    test_mlogloss = history.evals_result()['validation_1']['mlogloss']

    # Plotting error and log-loss curves
    plt.figure(figsize=(14, 6))

    # Error plot
    plt.subplot(1, 2, 1)
    plt.plot(train_error, label='Train Error')
    plt.plot(test_error, label='Test Error')
    plt.xlabel('Iterations')
    plt.ylabel('Error Rate')
    plt.title('Error vs Iterations')
    plt.legend()

    # Log-loss plot
    plt.subplot(1, 2, 2)
    plt.plot(train_mlogloss, label='Train Log-Loss')
    plt.plot(test_mlogloss, label='Test Log-Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Log-Loss')
    plt.title('Log-Loss vs Iterations')
    plt.legend()

    # Show the plots
    plt.tight_layout()
    plt.show()


def tabPFN(df):
    label_encoder =  LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['label'])
    
    # Features and labels
    X = df.drop(["label","sperm_id","displacement","time_elapsed","mad","wob","straightness","bcf","angular_displacement","curvature"], axis=1).values.astype(np.float32)
    #X = df.drop(["label","sperm_id"], axis=1).values.astype(np.float32)
    y = df["label"]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    '''
    # Apply RandomUnderSampler to balance
    rus = RandomUnderSampler(random_state=42)
    X_train_resampled, y_train_resampled = rus.fit_resample(X_train, y_train)
    '''
    # Apply SMOTE to balance
    smote = SMOTE(k_neighbors=2, random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        
    # Initialize a classifier
    clf_base =  TabPFNClassifier(
        ignore_pretraining_limits=True,
        inference_config = {"SUBSAMPLE_SAMPLES": 10000} # Needs to be set low so that not OOM on fitting intermediate nodes
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


    dump(tabpfn_tree_clf, "../models/TabPFN_4c_15s_extended.joblib")
    
    
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

if __name__ == "__main__":
    # Load the tracking data from a CSV file
    #df = pd.read_csv('../results/data_features_labelling_preprocessing/dataset_4c_30s_preprocessing_v2.csv')
    
    # Load digits dataset
    #digits = load_digits()
    #df = pd.DataFrame(data=digits.data)
    #df['target'] = digits.target

    # Filter to 4 classes (digits 0–3)
    #df = df[df['target'] < 4]
    
    
    #random_forest()
    #logistic_regression(df)
    XGBoost()
    #simple_NN(df)
    #tabPFN(df)
    #tabPFN_load()