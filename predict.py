import os
import argparse
import pickle
import numpy as np
import pandas as pd

def load_ml_model():
    model_path = os.path.join('models', 'ml_model.pkl')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"ML Model file not found at {model_path}. Run train_ml.py first.")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def load_dl_model():
    model_path = os.path.join('models', 'dl_model.pkl')
    preprocessor_path = os.path.join('models', 'dl_preprocessor.pkl')
    
    if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
        raise FileNotFoundError("DL Model or Preprocessor not found. Run train_dl.py first.")
        
    with open(preprocessor_path, 'rb') as f:
        preprocessor = pickle.load(f)
        
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        
    return model, preprocessor

def run_prediction(args):

    sin_time = np.sin(2 * np.pi * args.hour / 24.0)
    cos_time = np.cos(2 * np.pi * args.hour / 24.0)
    
 
    input_data = pd.DataFrame([{
        'Vehicle_Density': args.density,
        'Average_Speed': args.speed,
        'Traffic_Flow_Rate': args.flow,
        'Weather_Condition': args.weather,
        'Temperature': args.temp,
        'Visibility': args.visibility,
        'Signal_Strength': args.signal,
        'Latency': args.latency,
        'Packet_Loss': args.packet_loss,
        'Hour_of_Day': args.hour,
        'Day_of_Week': args.day,
        'Is_Peak_Hour': args.peak_hour,
        'Sin_Time': sin_time,
        'Cos_Time': cos_time
    }])
    
    classes = ['Low', 'Medium', 'High']
    
    print("\n--- Input Parameters ---")
    for k, v in input_data.iloc[0].to_dict().items():
        print(f"{k}: {v}")
        
    if args.model == 'ml':
        print("\nUsing traditional Machine Learning model...")
        pipeline = load_ml_model()
        
        pred = pipeline.predict(input_data)[0]
        probs = pipeline.predict_proba(input_data)[0]
        
        result_label = classes[pred]
        print(f"\nPrediction Results:")
        print(f"Predicted Traffic Condition: {result_label}")
        print("Class Probabilities:")
        for cls, prob in zip(classes, probs):
            print(f"  {cls}: {prob*100:.2f}%")
            
    elif args.model == 'dl':
        print("\nUsing Neural Network Deep Learning MLP model...")
        model, preprocessor = load_dl_model()
        
       
        processed_data = preprocessor.transform(input_data)
        if hasattr(processed_data, 'toarray'):
            processed_data = processed_data.toarray()
            
        pred = model.predict(processed_data)[0]
        probs = model.predict_proba(processed_data)[0]
            
        result_label = classes[pred]
        print(f"\nPrediction Results:")
        print(f"Predicted Traffic Condition: {result_label}")
        print("Class Probabilities:")
        for cls, prob in zip(classes, probs):
            print(f"  {cls}: {prob*100:.2f}%")
            
    else:
        print("Invalid model selection. Choose 'ml' or 'dl'.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict Traffic Condition using ML or DL")
    

    parser.add_argument('--model', type=str, default='ml', choices=['ml', 'dl'], help="Model type: 'ml' or 'dl'")
    
   
    parser.add_argument('--density', type=int, default=80, help="Vehicle density (vehicles/km)")
    parser.add_argument('--speed', type=float, default=45.0, help="Average vehicle speed (km/h)")
    parser.add_argument('--flow', type=float, default=300.0, help="Traffic flow rate (vehicles/h)")
    parser.add_argument('--weather', type=str, default='Clear', choices=['Clear', 'Rainy', 'Foggy'], help="Weather condition")
    parser.add_argument('--temp', type=float, default=25.0, help="Temperature in Celsius")
    parser.add_argument('--visibility', type=float, default=300.0, help="Visibility in meters")
    parser.add_argument('--signal', type=float, default=80.0, help="Network signal strength (dBm or score)")
    parser.add_argument('--latency', type=float, default=40.0, help="Network latency (ms)")
    parser.add_argument('--packet-loss', type=float, default=0.1, help="Network packet loss percentage")
    parser.add_argument('--hour', type=int, default=12, help="Hour of Day (0-23)")
    parser.add_argument('--day', type=str, default='Mon', choices=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], help="Day of Week")
    parser.add_argument('--peak-hour', type=int, default=0, choices=[0, 1], help="Is Peak Hour (0 or 1)")
    
    args = parser.parse_args()
    
    try:
        run_prediction(args)
    except Exception as e:
        print(f"\nError: {e}")
