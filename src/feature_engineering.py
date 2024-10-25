import pandas as pd
from sklearn.preprocessing import LabelEncoder

def encode_features(data):
    """Encode categorical variables using label encoding."""
    categorical_columns = ['Marital Status', 'Education Level', 'Smoking Status', 'Employment Status', 'Sleep Patterns', 'Dietary Habits']
    le = LabelEncoder()
    for col in categorical_columns:
        if col in data.columns:
            data[col] = le.fit_transform(data[col])
    return data

def select_features(data):
    """Select relevant features for modeling."""
    # Encode features first
    data = encode_features(data)
    
    features = ['Age', 'Income', 'Education Level', 'Family History of Depression', 'Sleep Patterns']
    target = 'History of Mental Illness'
    X = data[features]
    y = data[target].map({'No': 0, 'Yes': 1})  # Ensure target is numeric
    return X, y
