import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt

# Define the directories for data and images
DATA_DIR = './data/'
IMAGES_DIR = './images/'

# Ensure the images directory exists
os.makedirs(IMAGES_DIR, exist_ok=True)

# Load the dataset from the specified path
def load_data(path=os.path.join(DATA_DIR, 'depression_data.csv')):
    """Load the dataset from the specified path."""
    try:
        data = pd.read_csv(path)
        print(f"Data successfully loaded from {path}")
    except FileNotFoundError:
        print(f"File not found at {path}. Please ensure the file path is correct.")
        return None
    return data

# Data cleaning: Handle missing values and convert types if necessary
def clean_data(data):
    """Clean the dataset by handling missing values."""
    if data.isnull().sum().any():
        print("Missing values detected. Dropping missing values...")
        data_cleaned = data.dropna()
    else:
        print("No missing values detected.")
        data_cleaned = data
    return data_cleaned

def plot_correlation_matrix(X):
    """
    Plot and save the correlation matrix for the feature set X.
    """
    correlation_matrix = X.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Matrix of Features")
    plt.savefig(os.path.join(IMAGES_DIR, "correlation_matrix.png"))
    plt.close()

    # Print highly correlated pairs
    high_corr_pairs = [(feature1, feature2) for feature1 in correlation_matrix.columns 
                       for feature2 in correlation_matrix.columns 
                       if feature1 != feature2 and abs(correlation_matrix.loc[feature1, feature2]) > 0.8]
    print("Highly correlated feature pairs (correlation > 0.8):")
    for pair in high_corr_pairs:
        print(pair)

def exploratory_analysis(data):
    """Perform exploratory data analysis on the dataset."""
    print("Performing exploratory data analysis...")
    print("Basic Statistics:")
    print(data.describe())
    
    plot_target_variable_distribution(data)
    plot_numerical_feature_distributions(data)
    plot_categorical_feature_counts(data)
    detect_outliers(data)
    plot_correlation_with_target(data)

def plot_target_variable_distribution(data):
    plt.figure(figsize=(6, 4))
    sns.countplot(data=data, x='History of Mental Illness')
    plt.title("Distribution of Target Variable: History of Mental Illness")
    plt.xlabel("History of Mental Illness (No=0, Yes=1)")
    plt.ylabel("Count")
    plt.savefig(os.path.join(IMAGES_DIR, "target_variable_distribution.png"))
    plt.close()

def plot_numerical_feature_distributions(data):
    numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
    data[numerical_cols].hist(figsize=(14, 12), bins=20, color='teal', edgecolor='black')
    plt.suptitle("Distribution of Numerical Features")
    plt.savefig(os.path.join(IMAGES_DIR, "numerical_feature_distributions.png"))
    plt.close()

def plot_categorical_feature_counts(data):
    categorical_cols = data.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        plt.figure(figsize=(10, 5))
        sns.countplot(data=data, x=col, hue='History of Mental Illness')
        plt.title(f"{col} Distribution by Mental Illness History")
        plt.xticks(rotation=45)
        plt.savefig(os.path.join(IMAGES_DIR, f"{col}_distribution.png"))
        plt.close()

def detect_outliers(data):
    numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
    for col in numerical_cols:
        plt.figure(figsize=(6, 4))
        sns.boxplot(data=data, y=col)
        plt.title(f"Outlier Detection: {col}")
        plt.savefig(os.path.join(IMAGES_DIR, f"{col}_outliers.png"))
        plt.close()

def plot_correlation_with_target(data):
    target_corr = data.corr()['History of Mental Illness'].drop('History of Mental Illness').sort_values(ascending=False)
    plt.figure(figsize=(8, 6))
    target_corr.plot(kind='bar', color='salmon')
    plt.title("Correlation of Numerical Features with Target Variable")
    plt.ylabel("Correlation")
    plt.savefig(os.path.join(IMAGES_DIR, "correlation_with_target.png"))
    plt.close()

if __name__ == "__main__":
    data = load_data()
    
    if data is not None:
        cleaned_data = clean_data(data)
        cleaned_data.to_csv(os.path.join(DATA_DIR, 'cleaned_data.csv'), index=False)
        print(f"Cleaned data saved to '{os.path.join(DATA_DIR, 'cleaned_data.csv')}'")
        
        exploratory_analysis(cleaned_data)
