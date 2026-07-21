import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, send_from_directory

app = Flask(__name__)


def load_ml_pipeline():
    path = os.path.join('models', 'ml_model.pkl')
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        pipeline = pickle.load(f)
    return pipeline

def load_dl_model_and_preprocessor():
    model_path = os.path.join('models', 'dl_model.pkl')
    preprocessor_path = os.path.join('models', 'dl_preprocessor.pkl')
    if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
        return None, None
        
    with open(preprocessor_path, 'rb') as f:
        preprocessor = pickle.load(f)
        
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        
    return model, preprocessor

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/plots/<path:filename>')
def serve_plot(filename):
    return send_from_directory('plots', filename)

@app.route('/api/simulation-data')
def get_simulation_data():
   
    weather = np.random.choice(['Clear', 'Rainy', 'Foggy'], p=[0.7, 0.2, 0.1])
    day = np.random.choice(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    hour = np.random.randint(0, 24)
    is_peak = 1 if hour in [8, 9, 17, 18] else 0
    
  
    if is_peak:
        density = np.random.randint(140, 200)
        speed = np.random.uniform(15.0, 35.0)
    else:
        density = np.random.randint(10, 110)
        speed = np.random.uniform(40.0, 95.0)
        

    flow = density * speed * 0.15 + np.random.uniform(-20, 20)
    flow = max(10.0, flow)
    
  
    signal = np.random.uniform(40, 98)
    latency = np.random.uniform(10, 180)
    packet_loss = np.random.uniform(0.0, 4.5)
    
    return jsonify({
        'density': int(density),
        'speed': round(speed, 2),
        'flow': round(flow, 2),
        'weather': weather,
        'temp': round(np.random.uniform(15.0, 38.0), 1),
        'visibility': round(np.random.uniform(50.0, 500.0) if weather != 'Clear' else 500.0, 1),
        'signal': round(signal, 1),
        'latency': round(latency, 1),
        'packet_loss': round(packet_loss, 2),
        'hour': int(hour),
        'day': day,
        'peak_hour': is_peak
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        model_type = data.get('model_type', 'ml')
        
        
        density = float(data['density'])
        speed = float(data['speed'])
        flow = float(data['flow'])
        weather = data['weather']
        temp = float(data['temp'])
        visibility = float(data['visibility'])
        signal = float(data['signal'])
        latency = float(data['latency'])
        packet_loss = float(data['packet_loss'])
        hour = int(data['hour'])
        day = data['day']
        peak_hour = int(data['peak_hour'])
        
     
        sin_time = np.sin(2 * np.pi * hour / 24.0)
        cos_time = np.cos(2 * np.pi * hour / 24.0)
        
        
        input_df = pd.DataFrame([{
            'Vehicle_Density': density,
            'Average_Speed': speed,
            'Traffic_Flow_Rate': flow,
            'Weather_Condition': weather,
            'Temperature': temp,
            'Visibility': visibility,
            'Signal_Strength': signal,
            'Latency': latency,
            'Packet_Loss': packet_loss,
            'Hour_of_Day': hour,
            'Day_of_Week': day,
            'Is_Peak_Hour': peak_hour,
            'Sin_Time': sin_time,
            'Cos_Time': cos_time
        }])
        
        classes = ['Low', 'Medium', 'High']
        
        if model_type == 'ml':
            pipeline = load_ml_pipeline()
            if pipeline is None:
                return jsonify({'error': 'Machine Learning model not trained or available.'}), 500
                
            pred = int(pipeline.predict(input_df)[0])
            probs = pipeline.predict_proba(input_df)[0].tolist()
            
        elif model_type == 'dl':
            model, preprocessor = load_dl_model_and_preprocessor()
            if model is None:
                return jsonify({'error': 'Deep Learning model not trained or available.'}), 500
                
            processed = preprocessor.transform(input_df)
            if hasattr(processed, 'toarray'):
                processed = processed.toarray()
                
            pred = int(model.predict(processed)[0])
            probs = model.predict_proba(processed)[0].tolist()
        else:
            return jsonify({'error': 'Invalid model_type specified.'}), 400
            
        return jsonify({
            'prediction': classes[pred],
            'confidence': round(probs[pred] * 100, 2),
            'probabilities': {cls: round(prob * 100, 2) for cls, prob in zip(classes, probs)},
            'model_used': model_type.upper()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':

    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    app.run(debug=True, port=5000)
