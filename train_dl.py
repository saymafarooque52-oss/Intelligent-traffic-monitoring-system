import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score

def train_deep_learning():
    print("=== Starting Deep Learning (Neural Network MLP) Training ===")
    
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
    target_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
    y = df[target_col].map(target_mapping).values
    
    # Train-test split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Fit preprocessor
    numeric_features = [
        'Vehicle_Density', 'Average_Speed', 'Traffic_Flow_Rate', 
        'Temperature', 'Visibility', 'Signal_Strength', 
        'Latency', 'Packet_Loss', 'Hour_of_Day', 
        'Sin_Time', 'Cos_Time', 'Is_Peak_Hour'
    ]
    categorical_features = ['Weather_Condition', 'Day_of_Week']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )
    
    # Fit and transform
    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
    
    # Handle sparse matrix if any
    if hasattr(X_train_proc, 'toarray'):
        X_train_proc = X_train_proc.toarray()
        X_val_proc = X_val_proc.toarray()
        
    # Save DL preprocessor
    os.makedirs('models', exist_ok=True)
    preprocessor_path = os.path.join('models', 'dl_preprocessor.pkl')
    with open(preprocessor_path, 'wb') as f:
        pickle.dump(preprocessor, f)
    print(f"Saved DL preprocessor to {preprocessor_path}")
    
    # Initialize MLP Neural Network Classifier
    # This acts as our Multi-Layer Perceptron deep learning model.
    # We configure 2 hidden layers with 64 and 32 neurons respectively, ReLU activations,
    # and Adam optimizer with L2 weight decay.
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        alpha=1e-4,
        batch_size=64,
        learning_rate_init=0.003,
        max_iter=100,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.2,
        verbose=True
    )
    
    print("\nTraining MLP Neural Network...")
    model.fit(X_train_proc, y_train)
    
    # Predict and evaluate on test validation set
    y_pred = model.predict(X_val_proc)
    val_acc = accuracy_score(y_val, y_pred)
    print(f"\nNeural Network Validation Accuracy: {val_acc*100:.2f}%")
    print("Classification Report:")
    print(classification_report(y_val, y_pred, target_names=['Low', 'Medium', 'High']))
    
    # Save model
    model_path = os.path.join('models', 'dl_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Saved trained DL model to {model_path}")
    
    # Save training history curves
    os.makedirs('plots', exist_ok=True)
    plt.figure(figsize=(12, 5))
    
    # Plot loss curve
    plt.subplot(1, 2, 1)
    plt.plot(model.loss_curve_, label='Training Loss', color='#e74c3c', linewidth=2)
    plt.title('Neural Network Training Loss Decay')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot validation score (accuracy) curve
    plt.subplot(1, 2, 2)
    plt.plot(model.validation_scores_, label='Val Score (Acc)', color='#2ecc71', linewidth=2)
    plt.title('Validation Accuracy Curve')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('plots/dl_training_history.png', dpi=300)
    plt.close()
    print("Saved plots/dl_training_history.png")
    
    print("=== Neural Network MLP DL Training Completed ===")

if __name__ == '__main__':
    train_deep_learning()
