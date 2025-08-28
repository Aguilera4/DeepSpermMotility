import seaborn as sns
from sort.sort import *
import pandas as pd
import matplotlib.pyplot as plt
from classify_by_movement import *
import pandas as pd
from functions_features import *
from joblib import dump
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from tensorflow import keras
from tensorflow.keras import layers, optimizers
from sklearn.model_selection import learning_curve
import numpy as np
import warnings
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from tabpfn import TabPFNClassifier
from tabpfn_extensions.rf_pfn import (
    RandomForestTabPFNClassifier,
)
import joblib
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from imblearn.over_sampling import ADASYN

matplotlib.use("TkAgg")  # Use Tkinter-based backend
warnings.filterwarnings("ignore")


def draw_confusion_matrix(y_test,y_pred,title):
    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    # Create a heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Progressive', 'Non-progressive'], 
                yticklabels=['Progressive', 'Non-progressive'])

    # Add labels and title
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title(title)

    # Show the plot
    plt.show()
    
    
def draw_roc_auc_curve(y_test,y_pred_prob):
        # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)

    # Plot ROC Curve
    plt.figure(figsize=(10, 6))

    plt.subplot(1, 3, 1)
    plt.plot(fpr, tpr, color='b', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')

    # Plot Precision-Recall Curve
    plt.subplot(1, 3, 2)
    plt.plot(recall, precision, color='b', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    
    # Show all plots
    plt.tight_layout()
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
    epochs = len(results['validation_0']['logloss'])
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
    plt.plot(x_axis, results['validation_0']['logloss'], label='Train')
    plt.plot(x_axis, results['validation_1']['logloss'], label='Test')
    plt.xlabel('Epochs')
    plt.ylabel('Log Loss')
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
    plt.title("Learning curve - Random Forest")
    plt.xlabel("Training set size")
    plt.ylabel("Accuracy")

    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Test")

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


def feature_engineer(df,balanced_method,use_feature_selection,test_zize=0.3):
    # Features and labels
    X = df.drop(["label"], axis=1)
    y = LabelEncoder().fit_transform(df['label'])

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_zize, random_state=42, stratify=y)
    
    # Balanced method
    if balanced_method == 'SMOTE':
        # Apply SMOTE to balance
        smote = SMOTE(k_neighbors=2, random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    elif balanced_method == 'ADASYN':
        # Apply SMOTE to balance
        smote = ADASYN(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    else:
        X_train_resampled = X_train.copy()  
        y_train_resampled = y_train.copy() 
    
    # Feature selection
    X_train_selected = X_train_resampled.copy()  
    X_test_selected = X_test.copy()  
    if use_feature_selection == True:
        # Features importance
        X_train_selected, X_test_selected = select_features_importance(X_train_resampled,y_train_resampled,X_test)

    return [X, y, X_train_selected, X_test_selected, y_train_resampled, y_test]


############################## MODELS ##############################

def random_forest(X_train, X_test, y_train, y_test):
    # Train a Random Forest classifier
    clf = RandomForestClassifier(n_estimators=10, max_depth=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = clf.predict(X_test)
    
    # Show metrics
    show_metrics(y_test,y_pred)
    draw_confusion_matrix(y_test,y_pred)
    
    # cross-validation
    scores = cross_val_score(clf, X, y, cv=5)
    print("Fold accuracy:", np.round(scores, 3))
    print("Mean accuracy:", np.round(scores.mean(), 3))
    
    dump(clf, "../models/random_forest_2c.joblib")


def logistic_regression(X_train, X_test, y_train, y_test):
    # Train a Logistic regression model
    clf = LogisticRegression(solver='lbfgs', max_iter=100)
    clf.fit(X_train, y_train)

    # Get predictions and prediction probabilities
    y_pred = clf.predict(X_test)
    y_pred_prob = clf.predict_proba(X_test)[:, 1]  # Probability estimates for the positive class
    
    # Show metrics
    show_metrics(y_test,y_pred)
    draw_confusion_matrix(y_test,y_pred)
    draw_roc_auc_curve(y_test,y_pred_prob)
    
    dump(clf, "../models/linear_regression_2c.joblib")
    

def XGBoost(X_train, X_test, y_train, y_test):
    # Initialize XGBoost classifier
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric=["logloss", "auc"],
        learning_rate=0.05,
        min_child_weight=5,
        max_depth=3,    
        subsample=0.7,
        colsample_bytree=0.7,      
        gamma=1,
        reg_alpha=1,
        reg_lambda=1, 
        n_estimators=100,
        use_label_encoder=False,
        random_state=42
    )

    # eval_set
    eval_set = [(X_train, y_train), (X_test, y_test)]
    
    # Train the model
    model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

    # Predict
    y_pred = model.predict(X_test)

    # Show metrics
    show_metrics(y_test,y_pred)
    draw_confusion_matrix(y_test,y_pred)
    show_learning_curve(model.evals_result())
    
    dump(model, "../models/XGBoost_2c.joblib")


def tabPFN(X_train, X_test, y_train, y_test):
    
    # Initialize a classifier
    clf_base =  TabPFNClassifier(ignore_pretraining_limits=True,inference_config = {"SUBSAMPLE_SAMPLES": 1000})
    
    tabpfn_tree_clf = RandomForestTabPFNClassifier(
        tabpfn=clf_base,
        verbose=1,
        max_predict_time=60, # Will fit for one minute
        fit_nodes=True, # Wheather or not to fit intermediate nodes
        adaptive_tree=True, # Whather or not to validate if adding a leaf helps or not
    )

    # Train the model
    tabpfn_tree_clf.fit(X_train, y_train)

    # Predict labels
    y_pred = tabpfn_tree_clf.predict(X_test) 
    
    # Show metrics
    show_metrics(y_test,y_pred)
    draw_confusion_matrix(y_test,y_pred,'Confusion matrix (TabPFN)')
    
    # Predict probabilities
    y_pred_proba = tabpfn_tree_clf.predict_proba(X_test)
    draw_roc_auc_curve(y_test,y_pred_proba)


    dump(tabpfn_tree_clf, "../models/TabPFN_2c.joblib")
    
    
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
    
    

def simple_NN(X_train, X_test, y_train, y_test):
    
    model = keras.Sequential([
        layers.InputLayer(input_shape=(X_train.shape[1],)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    
    # Compile model
    model.compile(optimizer=optimizers.Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    # Train the model normally
    history = model.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])
    
    # Predict
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    
    plot_learning_curve_NN(history)
    
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    
    print(classification_report(y_test, y_pred))
    
    # Show metrics
    show_metrics(y_test,y_pred)
    draw_confusion_matrix(y_test,y_pred)
    
    dump(model, "../models/simple_NN_2c.joblib")

if __name__ == "__main__":
    # Load the tracking data from a CSV file
    df = pd.read_csv('../results/data_features_labelling_preprocessing/dataset_30s_2c.csv')
    
    X, y, X_train, X_test, y_train, y_test = feature_engineer(df=df,balanced_method='SMOTE',use_feature_selection=True,test_zize=0.3) #SMOTE ADASYN NO
    
    print("*** Data split ***")
    print("Train distribution")
    print(pd.Series(y_train).value_counts())
    print("Test distribution")
    print(pd.Series(y_test).value_counts())
    
    print("\n*** Random Forest ***")
    random_forest(X_train, X_test, y_train, y_test)
    
    print("\n*** Logistic regression ***")
    logistic_regression(X_train, X_test, y_train, y_test)
    
    print("\n*** XGBoost ***")
    XGBoost(X_train, X_test, y_train, y_test)
    
    print("\n*** TabPFN ***")
    tabPFN(X_train, X_test, y_train, y_test)
    
    print("\n*** NN ***")
    simple_NN(X_train, X_test, y_train, y_test)