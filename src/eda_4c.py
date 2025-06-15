import seaborn as sns
from sort.sort import *
import pandas as pd
import matplotlib.pyplot as plt
from classify_by_movement import *
import pandas as pd
from functions_features import *
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

matplotlib.use("TkAgg")  # Use Tkinter-based backend
    
def draw_class_distribution(df):
    # Count the number of samples for each class
    label_counts = df["label"].value_counts()
    
    print(label_counts)

    # Plot the label distribution
    plt.figure(figsize=(8, 6))
    sns.barplot(x=label_counts.index, y=label_counts.values, hue=label_counts.index, palette="viridis", legend=False)
    plt.title("Label Distribution", fontsize=16)
    plt.xlabel("Label (0=Rapidly progressive, 1=Slowly progressive, 2=Non-pogressive, 3=Inmotile)", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    plt.xticks([0, 1, 2, 3], ['Rapidly progressive', 'Slowly progressive', 'Non-pogressive', 'Inmotile'])  # Replace 0 and 1 with meaningful labels
    plt.show()
        
def draw_correlation_matix(df):
    # Calculate the correlation matrix
    corr_matrix = df.corr()

    # Plot the correlation matrix as a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Matrix", fontsize=16)
    plt.show()

def draw_null_count(df):
    # Get the number of null values in each column
    null_counts = df.isnull().sum()

    # Create a bar plot using Seaborn
    plt.figure(figsize=(8, 6))
    sns.barplot(x=null_counts.index, y=null_counts.values)
    
    # Rotate x-axis labels
    plt.xticks(rotation=45)

    # Add labels and title
    plt.title('Number of Null Values in Each Column')
    plt.xlabel('Columns')
    plt.ylabel('Number of Null Values')

    # Display the plot
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
    
    
def get_more_important_features(df):
    label_encoder =  LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['label'])
    
    # Features and labels
    X = df.drop(["label"], axis=1).values
    y = df["label"]
    
    # Initialize XGBoost classifier
    model = xgb.XGBClassifier(
        objective="multi:softmax",
        num_class=label_encoder.classes_.shape[0],
        eval_metric="mlogloss",
        use_label_encoder=False,
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )

    # Train the model
    model.fit(X, y, verbose=False)
    
    # Get feature importance
    feature_importances = model.feature_importances_

    # Display feature importances
    feature_importance_df = pd.DataFrame({
        'Feature': df.columns[:-1],
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)

    print(feature_importance_df)

    # Visualize the feature importances
    plt.barh(df.columns[:-1], feature_importances)
    plt.xlabel('Feature Importance')
    plt.title('Random Forest Feature Importance')
    plt.show()
        

if __name__ == "__main__":
    # Load the tracking data from a CSV file
    df = pd.read_csv('../results/data_features_labelling_preprocessing/dataset_30s_4c.csv')
    
    # Basic information
    print(df.head())
    print(df.info())
    print(df.describe())
    
    # Advanced information
    draw_class_distribution(df)
    #draw_correlation_matix(df)
    #draw_null_count(df)
    #draw_distribucion_columns(df)
    #show_outliers(df)
    get_more_important_features(df)