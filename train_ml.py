import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from xgboost import XGBClassifier

def train_machine_learning():
    print("=== Starting Traditional Machine Learning Training ===")
    
    # Load dataset
    csv_path = 'Traffic_Flow_Dataset.csv'
    if not os.path.exists(csv_path):
        print(f"Error: Dataset not found at {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    
    # Define features and target
    target_col = 'Traffic_Condition'
    feature_cols = [
        'Vehicle_Density', 'Average_Speed', 'Traffic_Flow_Rate', 
        'Weather_Condition', 'Temperature', 'Visibility', 
        'Signal_Strength', 'Latency', 'Packet_Loss', 
        'Hour_of_Day', 'Day_of_Week', 'Is_Peak_Hour', 
        'Sin_Time', 'Cos_Time'
    ]
    
    # Separate features and target
    X = df[feature_cols]
    # Ordinal mapping for target
    target_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
    y = df[target_col].map(target_mapping)
    
    if y.isnull().any():
        print("Warning: Missing or unexpected values in target column.")
        y = y.fillna(1)  # Fallback
        
    # Split dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")
    
    # Identify numeric and categorical features
    numeric_features = [
        'Vehicle_Density', 'Average_Speed', 'Traffic_Flow_Rate', 
        'Temperature', 'Visibility', 'Signal_Strength', 
        'Latency', 'Packet_Loss', 'Hour_of_Day', 
        'Sin_Time', 'Cos_Time', 'Is_Peak_Hour'
    ]
    categorical_features = ['Weather_Condition', 'Day_of_Week']
    
    # Create preprocessing pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )
    
    # Create directory for models
    os.makedirs('models', exist_ok=True)
    
    # 1. Random Forest Classifier Pipeline
    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    
    print("\nTraining Random Forest Classifier...")
    rf_pipeline.fit(X_train, y_train)
    rf_preds = rf_pipeline.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_preds)
    print(f"Random Forest Accuracy: {rf_acc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, rf_preds, target_names=['Low', 'Medium', 'High']))
    
    # 2. XGBoost Classifier Pipeline
    xgb_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss', n_jobs=-1))
    ])
    
    print("\nTraining XGBoost Classifier...")
    xgb_pipeline.fit(X_train, y_train)
    xgb_preds = xgb_pipeline.predict(X_test)
    xgb_acc = accuracy_score(y_test, xgb_preds)
    print(f"XGBoost Accuracy: {xgb_acc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, xgb_preds, target_names=['Low', 'Medium', 'High']))
    
    # Save the best model
    if rf_acc >= xgb_acc:
        best_pipeline = rf_pipeline
        best_name = "Random Forest"
        best_acc = rf_acc
    else:
        best_pipeline = xgb_pipeline
        best_name = "XGBoost"
        best_acc = xgb_acc
        
    print(f"\nBest Model: {best_name} with accuracy {best_acc:.4f}")
    
    # Save preprocessor and model separately or as pipeline
    # We will save the fitted pipeline directly, as it makes inference much cleaner.
    model_path = os.path.join('models', 'ml_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(best_pipeline, f)
        
    # Save target mapping for later reference
    with open(os.path.join('models', 'target_mapping.pkl'), 'wb') as f:
        pickle.dump({'mapping': target_mapping, 'reverse_mapping': {v: k for k, v in target_mapping.items()}}, f)
        
    print(f"Saved best ML pipeline to {model_path}")
    print("=== Traditional ML Training Completed ===")

if __name__ == '__main__':
    train_machine_learning()
