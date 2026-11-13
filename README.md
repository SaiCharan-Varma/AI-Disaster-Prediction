# DisastraSense – AI Disaster Prediction & Real-Time Hazard Intelligence

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.x-green.svg)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**DisastraSense** is an AI-powered disaster prediction and real-time hazard intelligence platform. It combines machine learning predictive models with live meteorological data from **Open-Meteo**, global seismic streams from the **USGS**, and **OpenStreetMap** geocoding to deliver actionable early warnings for floods, earthquakes, and hurricanes.

---

## 🚀 Key Features

### 1. 🌊 Flood Hazard Assessment
- **Live Environmental Sync**: One-click sync with Open-Meteo to auto-fetch current temperature, humidity, 24h precipitation, elevation, and river discharge for any coordinates.
- **Machine Learning Inference**: Random Forest classification model calculating flood probability (0–100%), hazard severity, and civil defense evacuation advisories.
- **Topological & Social Context**: Evaluates soil absorption (clay, loam, sand), urban land cover, drainage resilience, and historical inundation.

### 2. 🌋 Earthquake Magnitude Estimation & Live Stream
- **Seismic Regression Model**: Forecasts expected Richter magnitude based on geographic and tectonic coordinates.
- **Real-Time USGS Stream**: Fetches active earthquakes globally directly from the USGS Earthquake Hazards Program with magnitudes, focal depths, and tsunami warnings.

### 3. 🌀 Hurricane Wind Speed & Classification
- **Storm Physics Model**: Evaluates directional wind quadrants (NE, SE, SW, NW) and storm trajectories to forecast maximum sustained wind speeds.
- **Saffir-Simpson Scale**: Automatically categorizes storms from Tropical Depressions up to Category 5 Catastrophic Hurricanes.

### 4. ⚡ REST API Suite & Interactive Explorer
- Full JSON REST endpoints for integration with emergency services and mobile applications.
- In-browser interactive API console to test requests, view HTTP status codes, latency, and copy JSON payloads.

---

## 📡 Live API Endpoints

All prediction endpoints support both `application/json` and `multipart/form-data`:

| Endpoint | Method | Source | Description |
| :--- | :---: | :--- | :--- |
| `/api/predict/flood` | `POST` | Machine Learning | Evaluates environmental and meteorological parameters to predict flood probability and severity. |
| `/api/predict/earthquake` | `POST` | Machine Learning | Estimates anticipated earthquake magnitude based on geographic coordinates. |
| `/api/predict/hurricane` | `POST` | Machine Learning | Predicts maximum sustained wind speed (knots/kmh) and Saffir-Simpson category. |
| `/api/weather` | `GET` | Open-Meteo API | Fetches live temperature, humidity, precipitation, elevation, and river discharge for any `lat` and `lon`. |
| `/api/geocode` | `GET` | OpenStreetMap | Search any city or landmark worldwide (e.g. `?q=Tokyo`) to resolve coordinates. |
| `/api/live/earthquakes` | `GET` | USGS Hazards | Real-time global earthquake stream with magnitude, depth, and tsunami flags. |
| `/api/docs` | `GET` | OpenAPI Spec | Programmatic OpenAPI documentation for all available endpoints. |

---

## 🛠️ Quick Start Guide

### 1. Clone the Repository
```bash
git clone https://github.com/SaiCharan-Varma/AI-Disaster-Prediction.git
cd AI-Disaster-Prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate ML Models (First time only)
```bash
python create_all_models.py
```
*This trains and compiles the Flood, Earthquake, and Hurricane models.*

### 4. Run the Server
```bash
python app.py
```

### 5. Open in Your Browser
Open your browser and navigate to:
```
http://127.0.0.1:5000/
```
- **Live Disaster AI Suite & API Explorer**: `http://127.0.0.1:5000/` (or `/predict`)
- **Multi-Panel Form View**: `http://127.0.0.1:5000/app`
- **Landing Overview**: `http://127.0.0.1:5000/home`
- **Interactive OpenAPI Documentation**: `http://127.0.0.1:5000/api/docs`

---

## 📂 Project Structure

```text
├── app.py                      # Flask REST API server & routing
├── create_all_models.py        # Model training script (Flood, Earthquake, Hurricane)
├── index.html                  # Live AI Command Center & API Explorer (served at /)
├── templates/
│   ├── index.html              # Template mirror for Flask rendering
│   ├── main.html               # Multi-panel prediction dashboard (/app)
│   └── home.html               # Landing overview (/home)
├── plots/                      # Model performance metrics & confusion matrices
├── atlantic.csv                # Historical hurricane dataset
├── earthquake (1).csv          # Global seismic dataset
├── requirements.txt            # Python package dependencies
└── visualize_flood_metrics.py  # Model evaluation visualization script
```

---

## 📄 License
This project is open-source under the MIT License.
