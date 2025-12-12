import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import joblib
import os

# Define paths
DATA_PATH = os.path.join("data", "Telco-Customer-Churn.csv")
MODEL_PATH = os.path.join("src", "model", "model.pkl")
PREPROCESSOR_PATH = os.path.join("src", "model", "preprocessor.pkl") # We can save just the pipeline if we wrap the model in it

def train():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    
    # Preprocessing
    # TotalCharges is object, need to convert to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(0, inplace=True)
    
    # Drop customerID
    df = df.drop('customerID', axis=1)
    
    # Encode Target
    df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # Define features
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_features = [c for c in X.columns if c not in numeric_features]
    
    print(f"Numeric features: {numeric_features}")
    print(f"Categorical features: {categorical_features}")
    
    # Build Pipeline
    # Numeric: Impute -> Scale
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical: Impute -> OneHot
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    # Full Pipeline
    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training model...")
    clf.fit(X_train, y_train)
    
    print("Evaluating...")
    score = clf.score(X_test, y_test)
    print(f"Test Accuracy: {score:.4f}")
    
    # Save
    if not os.path.exists(os.path.dirname(MODEL_PATH)):
        os.makedirs(os.path.dirname(MODEL_PATH))
        
    print(f"Saving model to {MODEL_PATH}")
    joblib.dump(clf, MODEL_PATH)
    # Note: We are saving the full pipeline including preprocessing into model.pkl
    # This makes inference easier as we just feed raw data
    
if __name__ == "__main__":
    train()
