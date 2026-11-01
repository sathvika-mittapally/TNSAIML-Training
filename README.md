# AQI Prediction Project

This repository contains a complete end‑to‑end pipeline for **Air Quality Index (AQI) prediction** using synthetic environmental data.

## Project Structure
```
AQI/
├─ data/                # Dataset generator and generated CSV
│   └─ generate_dataset.py
├─ models/              # Trained model files (created after training)
├─ src/                 # Source code
│   ├─ train_model.py   # Train a RandomForestRegressor and save the model
│   └─ app.py           # Flask API that serves AQI predictions
├─ .vscode/             # VS Code launch & task configurations
│   ├─ launch.json
│   └─ tasks.json
├─ requirements.txt     # Python dependencies
└─ README.md            # Project overview & usage instructions
```

## Prerequisites
- Windows 10/11 with **Python 3.14** (already on your system)
- **Visual Studio Code** installed
- Optional but recommended: the VS Code **Python** extension (`ms-python.python`)

## Quick‑Start Guide (run inside VS Code)
1. **Open the folder**
   - `File → Open Folder…` → choose `D:\AQI`
2. **Create a virtual environment** (VS Code terminal will automatically activate it when you open the folder)
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\activate
   ```
3. **Install dependencies**
   ```powershell
   pip install -r requirements.txt
   ```
4. **Generate the synthetic dataset**
   - Run the task *Generate Dataset* (`Ctrl+Shift+B` → *Generate Dataset*) **or** from the terminal:
   ```powershell
   python data\generate_dataset.py
   ```
   The file `data\air_quality_data.csv` will be created.
5. **Train the model**
   - Run the task *Train Model* (`Ctrl+Shift+B` → *Train Model*) **or**:
   ```powershell
   python src\train_model.py
   ```
   A model file `models\aqi_rf.pkl` will be saved.
6. **Run the API server**
   - Run the task *Run API* (`Ctrl+Shift+B` → *Run API*) **or**:
   ```powershell
   python src\app.py
   ```
   The server listens on `http://127.0.0.1:5000`.
7. **Test the endpoint** (you can use the VS Code *REST Client* extension or `curl`):
   ```powershell
   curl -X POST http://127.0.0.1:5000/predict \
        -H "Content-Type: application/json" \
        -d "{\"Temperature\": 28, \"Humidity\": 60, \"Wind_Speed\": 5, \"PM2.5\": 35, \"PM10\": 80, \"NO2\": 30, \"SO2\": 10, \"CO\": 0.8, \"O3\": 45}" 
   ```
   You will receive a JSON response with the predicted AQI.

## Debugging in VS Code
The provided `launch.json` contains pre‑configured **Python** debug configurations for:
- **Generate Dataset** (`data/generate_dataset.py`)
- **Train Model** (`src/train_model.py`)
- **Flask API** (`src/app.py`, with Flask's debugger enabled)

Press `F5` after selecting the desired configuration.

---
### FAQ
- **Where is the trained model stored?** → `models/aqi_rf.pkl`
- **Can I use a different algorithm?** → Edit `src/train_model.py` – the script is deliberately simple (RandomForestRegressor) but you can replace it with any scikit‑learn estimator.
- **How do I visualize results?** – Load `data/air_quality_data.csv` in a Jupyter notebook and plot `AQI` vs. any feature using `matplotlib` or `seaborn`.

Enjoy building and extending your AQI prediction system!
