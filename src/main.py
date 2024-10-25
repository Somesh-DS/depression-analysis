import os
import pandas as pd
from data_preprocessing import load_data, clean_data, exploratory_analysis, plot_correlation_matrix
from model_training import train_and_evaluate_models  # Update this import
from evaluation import plot_roc_curve, check_bias
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

# Set directory for data
DATA_DIR = '../data/'  # Adjust this path as necessary

# Define file paths
raw_data_path = os.path.join(DATA_DIR, "depression_data.csv")
cleaned_data_path = os.path.join(DATA_DIR, "cleaned_data.csv")
images_path = os.path.join("images")

# Ensure the necessary directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(images_path, exist_ok=True)

# Step 1: Data Preprocessing (Load and Clean Data)
print("Loading raw data...")
data = load_data(raw_data_path)

if data is not None:
    print("Cleaning the data...")
    cleaned_data = clean_data(data)

    cleaned_data.to_csv(cleaned_data_path, index=False)
    print(f"Cleaned data saved to '{cleaned_data_path}'")
    
    exploratory_analysis(cleaned_data)
else:
    print("Data loading failed. Exiting the program.")
    exit()

# Step 2: Model Training and Evaluation
print("Loading cleaned data for model training...")
try:
    data = pd.read_csv(cleaned_data_path)
except FileNotFoundError:
    print(f"Cleaned data file not found at {cleaned_data_path}. Exiting.")
    exit()

# Check for necessary columns
if 'History of Mental Illness' not in data.columns:
    print("Target variable 'History of Mental Illness' not found in the cleaned data. Exiting.")
    exit()

# Convert categorical target variable to numeric
data['History of Mental Illness'] = data['History of Mental Illness'].map({'No': 0, 'Yes': 1})

# Drop the Name column
data = data.drop(columns=['Name'], errors='ignore')

# Split features and target variable
X = data.drop('History of Mental Illness', axis=1)
y = data['History of Mental Illness']

# Check if the dataset is empty
if X.empty or y.empty:
    print("Feature set or target variable is empty after cleaning. Exiting.")
    exit()

# Encode categorical variables
categorical_cols_to_encode = X.select_dtypes(include=['object']).columns.tolist()
X_encoded = pd.get_dummies(X[categorical_cols_to_encode], drop_first=True)
X = X.drop(categorical_cols_to_encode, axis=1).join(X_encoded)

# Call the correlation matrix function here
print("Plotting the correlation matrix to check feature relationships...")
plot_correlation_matrix(X)

# Ensure all columns are numeric
X = X.apply(pd.to_numeric, errors='coerce')
X.dropna(inplace=True)
y = y[X.index]

# Class distribution before SMOTE
print("Class distribution before SMOTE:")
print(y.value_counts())

# Resampling
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Class distribution after SMOTE
print("Class distribution after SMOTE:")
print(pd.Series(y_resampled).value_counts())

# Feature scaling
scaler = StandardScaler()
X_resampled = scaler.fit_transform(X_resampled)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train and evaluate models
print("Training and evaluating models...")
lr_model, rf_model, xgb_model = train_and_evaluate_models(X_train, y_train, X_test, y_test)

# Step 3: Plot ROC curve and check bias
print("Generating ROC curve for XGBoost model...")
roc_curve_image_path = os.path.join(images_path, "roc_curve.png")
# Use the XGBoost model for the ROC curve
plot_roc_curve(xgb_model, X_test, y_test, save_path=roc_curve_image_path)
print(f"ROC curve saved at: {roc_curve_image_path}")

# Check for bias based on Gender if present in the dataset
if 'Gender' in data.columns:
    print("Checking for bias based on Gender...")
    bias_image_path = os.path.join(images_path, "gender_bias.png")
    check_bias(data, xgb_model, X_test, y_test, save_path=bias_image_path)
    print(f"Gender bias plot saved at: {bias_image_path}")
else:
    print("Gender column not found in the dataset for bias check.")
