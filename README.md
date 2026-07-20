# Intelligent Traffic Monitoring System

An end-to-end intelligent traffic monitoring and management system using machine learning and deep learning (Neural Network) models. The system evaluates real-time traffic features and sensor readings to classify traffic congestion levels (`Low`, `Medium`, `High`).

## Project Stack
- **Languages:** Python, JavaScript (ES6), HTML5, CSS3
- **Machine Learning & Deep Learning:** Scikit-Learn (Random Forest, MLP Neural Network), XGBoost
- **Data Engineering & Visualization:** Pandas, NumPy, Matplotlib, Seaborn
- **Web App / Backend:** Flask

---

## Repository Structure
- `Traffic_Flow_Dataset.csv`: Historical traffic flow dataset containing traffic features and congestion classes.
- `requirements.txt`: Python package dependencies.
- `eda.py`: Performs exploratory data analysis and generates visualization plots in the `plots/` directory.
- `train_ml.py`: Preprocesses the dataset and trains traditional Machine Learning models (Random Forest, XGBoost), saving the best model to the `models/` directory.
- `train_dl.py`: Preprocesses the dataset and trains a Multi-Layer Perceptron (MLP) Neural Network, saving weights and history.
- `predict.py`: CLI-level script to run custom manual predictions.
- `app.py`: Flask-based backend server.
- `templates/index.html`: Interactive, highly-styled glassmorphic web dashboard.
- `static/style.css`: Clean, dark-mode CSS variables, styles, and animation classes.
- `static/app.js`: Interactive elements, sliders, live sensor telemetry simulation, and inference API client.

---

## Dataset Description
The model is trained on `Traffic_Flow_Dataset.csv` containing the following attributes:
- **Vehicle_Density:** Number of vehicles per kilometer.
- **Average_Speed:** Average speed of the flowing vehicles (km/h).
- **Traffic_Flow_Rate:** Flow rate (vehicles/hour).
- **Weather_Condition:** Ambient weather conditions (`Clear`, `Rainy`, `Foggy`).
- **Temperature:** Ambient temperature in Celsius.
- **Visibility:** Line-of-sight visibility in meters.
- **Is_Peak_Hour:** Binary flag (0 or 1) indicating if the observation falls within peak hours.
- **Hour_of_Day / Day_of_Week:** Temporal variables.
- **Sin_Time / Cos_Time:** Continuous periodic encoding representing the hour of the day.
- **Signal_Strength / Latency / Packet_Loss:** Edge network connection parameters.
- **Traffic_Condition:** Target variable representing traffic status (`Low`, `Medium`, `High`).

---

## Setup & Installation

### 1. Initialize Virtual Environment
Create a virtual environment to manage dependencies:
```bash
python -m venv venv
```

Activate the environment:
- **Windows (PowerShell):**
  ```powershell
  venv\Scripts\Activate.ps1
  ```
- **Windows (CMD):**
  ```cmd
  venv\Scripts\activate.bat
  ```
- **macOS / Linux:**
  ```bash
  source venv/bin/activate
  ```

### 2. Install Packages
Install the required packages:
```bash
pip install -r requirements.txt
```

---

## Running the Pipelines

### Step 1: Run Data Exploratory Analysis
Execute the script to inspect the dataset statistics and output analysis plots:
```bash
python eda.py
```
This generates and saves charts in the `plots/` directory.

### Step 2: Train Machine Learning Models
Preprocess the dataset and train traditional classifier algorithms:
```bash
python train_ml.py
```
This trains Random Forest and XGBoost, selects the best model, and saves it to `models/ml_model.pkl`.

### Step 3: Train Deep Learning Neural Network
Preprocess and train the Multi-Layer Perceptron (MLP) Neural Network:
```bash
python train_dl.py
```
This trains a multi-hidden-layer feedforward neural network and saves the model to `models/dl_model.pkl` along with validation learning curves.

### Step 4: Run Command-Line Inference
You can test model prediction using command line parameters:
```bash
python predict.py --model dl --density 160 --speed 22 --flow 800 --weather Rainy --hour 8 --peak-hour 1
```

---

## Starting the Web Application Dashboard

Launch the Flask server to view the telemetry simulator and predict traffic using a web dashboard interface:
```bash
python app.py
```
Once launched, open your web browser and navigate to:
```
http://127.0.0.1:5000
```

### Dashboard Features:
1. **Live Sensor Telemetry Monitor:** Simulates incoming real-time sensor streams from roadside cameras and wireless edge nodes.
2. **AI Engine Decisions:** Runs live classification using the ML or DL engine, triggering status light colors.
3. **Interactive Inference Playground:** Adjust custom sliders to run custom predictions.
4. **Analytics View:** Switch tabs to inspect dataset correlation maps, distributions, and neural network training curves.
