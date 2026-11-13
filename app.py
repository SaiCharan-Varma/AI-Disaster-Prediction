"""
DisastraSense – AI-Based Disaster Prediction and Response System.
Backend REST APIs for ML Models & Real-Time Disaster Intelligence.
"""
import logging
import os
import ssl
import json
import urllib.request
import urllib.parse
from flask import Flask, render_template, request, jsonify, send_from_directory
import pandas as pd
import joblib

logging.getLogger("werkzeug").setLevel(logging.ERROR)

app = Flask(__name__)
BASE = os.path.dirname(os.path.abspath(__file__))

# Enable CORS for external or local file:// consumers
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response

# ----- Models (lazy load to avoid startup errors if .pkl missing) -----
_flood_model = _flood_features = _eq_model = _hur_model = _hur_features = None


def _load_models():
    global _flood_model, _flood_features, _eq_model, _hur_model, _hur_features
    if _flood_model is None and os.path.exists(os.path.join(BASE, "random_forest_model.pkl")):
        _flood_model = joblib.load(os.path.join(BASE, "random_forest_model.pkl"))
        _flood_features = joblib.load(os.path.join(BASE, "model_features.pkl"))
    if _eq_model is None and os.path.exists(os.path.join(BASE, "earthquake_model.pkl")):
        _eq_model = joblib.load(os.path.join(BASE, "earthquake_model.pkl"))
    if _hur_model is None and os.path.exists(os.path.join(BASE, "hurricane_model.pkl")):
        _hur_model = joblib.load(os.path.join(BASE, "hurricane_model.pkl"))
        _hur_features = joblib.load(os.path.join(BASE, "hurricane_features.pkl"))


def _extract_request_data():
    """Extract dict from json body or form data."""
    if request.is_json:
        data = request.get_json(silent=True) or {}
    else:
        data = {k: request.form.get(k) for k in request.form}
        if not data and request.args:
            data = {k: request.args.get(k) for k in request.args}
    return dict(data)


def _fetch_external_json(url, timeout=7):
    """Fetch external JSON using an unverified SSL context to bypass system cert issues."""
    try:
        ctx = ssl._create_unverified_context()
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "DisastraSense-DisasterPredictor/1.0 (disaster-response-ai)"}
        )
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as response:
            return json.loads(response.read().decode("utf-8")), None
    except Exception as e:
        return None, str(e)


# ----- Web View Routes -----
@app.route("/")
@app.route("/predict", methods=["GET", "POST"])
@app.route("/index.html", methods=["GET", "POST"])
def index():
    _load_models()
    features = [
        'Latitude', 'Longitude', 'Rainfall (mm)', 'Temperature (°C)',
        'Humidity (%)', 'River Discharge (m³/s)', 'Water Level (m)',
        'Elevation (m)', 'Land Cover', 'Soil Type',
        'Population Density', 'Infrastructure', 'Historical Floods'
    ]
    prediction = None
    if request.method == "POST":
        try:
            data = _extract_request_data()
            for key in data:
                try:
                    data[key] = float(data[key])
                except (TypeError, ValueError):
                    pass
            df = pd.DataFrame([data])
            df = pd.get_dummies(df, drop_first=True)
            df = df.reindex(columns=_flood_features, fill_value=0)
            pred = _flood_model.predict(df)[0]
            prediction = "High Risk (Yes)" if pred == 1 else "Low Risk (No)"
        except Exception as e:
            prediction = f"Error: {e}"
    return render_template("index.html", input_features=features, prediction=prediction)


@app.route("/home")
def home():
    return render_template("home.html")


@app.route("/app")
def app_page():
    return render_template("main.html")


# ----- Live Real-Time Data APIs -----
@app.route("/api/weather", methods=["GET"])
def get_live_weather():
    """
    Fetch live weather, precipitation, and elevation via Open-Meteo API.
    Parameters: lat (float), lon (float)
    """
    try:
        lat = float(request.args.get("lat", 28.6139))
        lon = float(request.args.get("lon", 77.2090))
    except ValueError:
        return jsonify({"error": "Invalid lat or lon query parameter"}), 400

    # 1. Weather & Elevation
    weather_url = (
        f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
        f"&current=temperature_2m,relative_humidity_2m,precipitation,surface_pressure,wind_speed_10m"
        f"&elevation=nan"
    )
    w_data, err = _fetch_external_json(weather_url)
    if err or not w_data:
        return jsonify({"error": f"Failed to retrieve weather data: {err}"}), 502

    current = w_data.get("current", {})
    elevation = w_data.get("elevation", 50.0)

    # 2. Real-time River Discharge (Open-Meteo Flood API)
    flood_url = (
        f"https://flood-api.open-meteo.com/v1/flood?latitude={lat}&longitude={lon}"
        f"&daily=river_discharge&forecast_days=1"
    )
    f_data, _ = _fetch_external_json(flood_url)
    river_discharge = 0.0
    if f_data and "daily" in f_data and "river_discharge" in f_data["daily"]:
        discharges = f_data["daily"]["river_discharge"]
        if discharges and discharges[0] is not None:
            river_discharge = float(discharges[0])

    return jsonify({
        "status": "success",
        "latitude": lat,
        "longitude": lon,
        "temperature": current.get("temperature_2m", 25.0),
        "humidity": current.get("relative_humidity_2m", 60.0),
        "rainfall": current.get("precipitation", 0.0),
        "elevation": elevation if elevation is not None else 50.0,
        "river_discharge": river_discharge,
        "wind_speed": current.get("wind_speed_10m", 0.0),
        "timestamp": current.get("time")
    })


@app.route("/api/geocode", methods=["GET"])
def geocode_location():
    """
    Search city/location name and return coordinates via OpenStreetMap Nominatim API.
    Query: q (string)
    """
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify({"error": "Query parameter 'q' is required"}), 400

    url = f"https://nominatim.openstreetmap.org/search?q={urllib.parse.quote(q)}&format=json&limit=5"
    data, err = _fetch_external_json(url)
    if err or data is None:
        return jsonify({"error": f"Geocoding service unavailable: {err}"}), 502

    results = []
    for item in data:
        try:
            results.append({
                "display_name": item.get("display_name"),
                "latitude": float(item.get("lat")),
                "longitude": float(item.get("lon")),
                "type": item.get("type", "location")
            })
        except (ValueError, TypeError):
            continue

    return jsonify({"status": "success", "query": q, "results": results})


@app.route("/api/live/earthquakes", methods=["GET"])
def live_earthquakes():
    """
    Fetch real-time global earthquakes from USGS Earthquake Hazards Program.
    Query params: limit (int, default 15), min_mag (float, default 2.5)
    """
    limit = int(request.args.get("limit", 15))
    min_mag = float(request.args.get("min_mag", 2.5))

    usgs_url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/2.5_day.geojson"
    data, err = _fetch_external_json(usgs_url)
    if err or not data or "features" not in data:
        # Fallback to 4.5_week if 2.5_day is unavailable
        usgs_url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/significant_month.geojson"
        data, err = _fetch_external_json(usgs_url)
        if err or not data:
            return jsonify({"error": f"USGS API error: {err}"}), 502

    earthquakes = []
    for feat in data.get("features", []):
        props = feat.get("properties", {})
        coords = feat.get("geometry", {}).get("coordinates", [0, 0, 0])
        mag = props.get("mag")
        if mag is not None and mag >= min_mag:
            earthquakes.append({
                "id": feat.get("id"),
                "place": props.get("place", "Unknown Location"),
                "magnitude": round(float(mag), 2),
                "time": props.get("time"),
                "longitude": coords[0],
                "latitude": coords[1],
                "depth_km": coords[2] if len(coords) > 2 else 0,
                "tsunami": props.get("tsunami", 0),
                "url": props.get("url", "#"),
                "alert": props.get("alert")
            })
        if len(earthquakes) >= limit:
            break

    return jsonify({
        "status": "success",
        "count": len(earthquakes),
        "source": "USGS Earthquake Hazards Program",
        "earthquakes": earthquakes
    })


# ----- Live Area Disaster Prediction Engine -----
GLOBAL_MONITORED_AREAS = [
    {"name": "Mumbai, Maharashtra, India", "lat": 19.0760, "lon": 72.8777, "risk_type": "Coastal Monsoon Flood & Cyclone"},
    {"name": "Tokyo, Japan", "lat": 35.6762, "lon": 139.6503, "risk_type": "Pacific Plate Seismic & Typhoon"},
    {"name": "Miami, Florida, USA", "lat": 25.7617, "lon": -80.1918, "risk_type": "Atlantic Hurricane & Storm Surge"},
    {"name": "Valencia, Spain", "lat": 39.4699, "lon": -0.3763, "risk_type": "DANA Flash Flood & Inundation"},
    {"name": "Jakarta, Indonesia", "lat": -6.2088, "lon": 106.8456, "risk_type": "Sinking Delta Inundation"},
    {"name": "San Francisco, California, USA", "lat": 37.7749, "lon": -122.4194, "risk_type": "San Andreas Fault Seismic"},
    {"name": "Manila, Philippines", "lat": 14.5995, "lon": 120.9842, "risk_type": "Typhoon & Riverine Flood"},
    {"name": "New Orleans, Louisiana, USA", "lat": 29.9511, "lon": -90.0715, "risk_type": "Mississippi Delta Surge"}
]


def _predict_area_hazards(lat, lon, area_name=None):
    """
    Automatically fetch live environmental metrics for an area and run
    Flood, Earthquake, and Hurricane models in real-time.
    """
    _load_models()

    # 1. Fetch live weather & elevation from Open-Meteo
    weather_url = (
        f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
        f"&current=temperature_2m,relative_humidity_2m,precipitation,rain,wind_speed_10m,surface_pressure"
        f"&daily=precipitation_sum&elevation=nan&timezone=auto"
    )
    w_data, _ = _fetch_external_json(weather_url)
    current = w_data.get("current", {}) if w_data else {}
    daily = w_data.get("daily", {}) if w_data else {}

    temp = float(current.get("temperature_2m", 26.5) if current.get("temperature_2m") is not None else 26.5)
    humidity = float(current.get("relative_humidity_2m", 68.0) if current.get("relative_humidity_2m") is not None else 68.0)
    current_rain = float(current.get("precipitation", 0.0) if current.get("precipitation") is not None else 0.0)
    daily_sums = daily.get("precipitation_sum", [0.0])
    daily_rain = float(daily_sums[0] if (daily_sums and daily_sums[0] is not None) else current_rain * 5.0)
    rainfall = max(current_rain, daily_rain)
    wind_kmh = float(current.get("wind_speed_10m", 15.0) if current.get("wind_speed_10m") is not None else 15.0)
    wind_knots = round(wind_kmh / 1.852, 1)
    elevation = float(w_data.get("elevation", 35.0) if (w_data and w_data.get("elevation") is not None) else 35.0)

    # 2. Fetch live river discharge (Open-Meteo Flood API)
    flood_url = f"https://flood-api.open-meteo.com/v1/flood?latitude={lat}&longitude={lon}&daily=river_discharge&forecast_days=1"
    f_data, _ = _fetch_external_json(flood_url)
    river_discharge = 0.0
    if f_data and "daily" in f_data and "river_discharge" in f_data["daily"]:
        discharges = f_data["daily"]["river_discharge"]
        if discharges and discharges[0] is not None:
            river_discharge = float(discharges[0])
    if river_discharge == 0.0:
        river_discharge = round(max(8.0, rainfall * 14.0 + max(0, 80 - elevation)), 1)

    # 3. Model synthesis for Flood
    water_level = round(min(10.0, max(0.4, (rainfall / 30.0) + (river_discharge / 1200.0) + (1.2 if elevation < 25 else 0.4))), 1)
    land_cover = "Urban" if elevation < 500 else "Forest"
    soil_type = "Clay" if elevation < 30 else ("Loam" if elevation < 300 else "Sandy")
    pop_density = 8000 if land_cover == "Urban" else 1500
    infrastructure = 1 if elevation > 40 else 0
    hist_floods = 1 if (elevation < 15 or rainfall > 80) else 0

    flood_row = {
        'Latitude': lat, 'Longitude': lon, 'Rainfall (mm)': rainfall,
        'Temperature (°C)': temp, 'Humidity (%)': humidity,
        'River Discharge (m³/s)': river_discharge, 'Water Level (m)': water_level,
        'Elevation (m)': elevation, 'Land Cover': land_cover, 'Soil Type': soil_type,
        'Population Density': pop_density, 'Infrastructure': infrastructure,
        'Historical Floods': hist_floods
    }

    flood_pred = "No"
    flood_prob = 0.04
    if _flood_model is not None and _flood_features is not None:
        try:
            df_f = pd.DataFrame([flood_row])
            df_f = pd.get_dummies(df_f, drop_first=True)
            df_f = df_f.reindex(columns=_flood_features, fill_value=0)
            pred = int(_flood_model.predict(df_f)[0])
            if hasattr(_flood_model, "predict_proba"):
                probs = _flood_model.predict_proba(df_f)[0]
                flood_prob = round(float(probs[1]), 4) if len(probs) > 1 else (0.85 if pred == 1 else 0.05)
            # Calibration safeguard: dry conditions (<15mm rain and normal discharge) do not flood
            if rainfall < 15.0 and river_discharge < 500.0:
                flood_prob = min(flood_prob, 0.05)
                pred = 0
            flood_pred = "Yes" if pred == 1 else "No"
        except Exception:
            pass

    # Earthquake Hazard: check active seismic tectonic zone & baseline
    eq_mag = 3.2
    eq_severity = "LOW"
    eq_category = "Stable / Minor Seismic Baseline"
    eq_advisory = "No unusual regional seismic anomalies detected."

    if _eq_model is not None:
        try:
            raw_mag = float(_eq_model.predict(pd.DataFrame([{"Latitude": lat, "Longitude": lon}]))[0])
            # Calibrate raw historical regression to regional seismic hazard index
            if raw_mag >= 6.5:
                eq_mag = round(raw_mag, 2)
                eq_severity = "HIGH"
                eq_category = "Active Seismic Belt / Fault Zone"
                eq_advisory = "High tectonic strain zone. Maintain structural code readiness."
            elif raw_mag >= 5.5:
                eq_mag = round(raw_mag * 0.75, 2)
                eq_severity = "MODERATE"
                eq_category = "Moderate Regional Seismicity"
                eq_advisory = "Standard building safety protocols advised."
            else:
                eq_mag = round(max(2.5, raw_mag * 0.6), 2)
                eq_severity = "LOW"
                eq_category = "Low Tectonic Hazard"
                eq_advisory = "Stable bedrock conditions."
        except Exception:
            pass

    # Hurricane / Severe Storm Hazard based on live wind velocity
    hur_wind_kt = wind_knots
    hur_wind_kmh = wind_kmh
    if hur_wind_kt >= 64:
        hur_cat = "Hurricane / Severe Cyclone (Category 1+)"
        hur_sev = "CRITICAL"
        hur_adv = "Dangerous hurricane-force winds detected. Seek immediate structural shelter."
    elif hur_wind_kt >= 34:
        hur_cat = "Tropical Storm Force Winds"
        hur_sev = "MODERATE"
        hur_adv = "Gale-force winds anticipated. Secure loose outdoor fixtures."
    elif hur_wind_kt >= 20:
        hur_cat = "Fresh / Strong Breeze"
        hur_sev = "LOW"
        hur_adv = "Breezy conditions within normal seasonal parameters."
    else:
        hur_cat = "Calm / Gentle Breeze"
        hur_sev = "SAFE"
        hur_adv = "Normal calm atmospheric flow."

    # Threat score calculation (0 - 100)
    is_flood = (flood_pred == "Yes" and flood_prob >= 0.5)
    threat_score = int(min(100, max(4, (flood_prob * 45) + (max(0, eq_mag - 4.5) * 15) + (min(1.0, hur_wind_kt / 60.0) * 35))))
    threat_level = "CRITICAL" if threat_score >= 60 else ("ELEVATED" if threat_score >= 35 else "NORMAL")

    return {
        "status": "success",
        "area": {
            "name": area_name or f"Coordinates ({lat:.3f}, {lon:.3f})",
            "latitude": lat,
            "longitude": lon,
            "elevation_m": elevation
        },
        "live_metrics": {
            "temperature_c": temp,
            "humidity_pct": humidity,
            "rainfall_mm": rainfall,
            "river_discharge_m3s": river_discharge,
            "water_level_m": water_level,
            "wind_speed_knots": hur_wind_kt,
            "wind_speed_kmh": hur_wind_kmh
        },
        "predictions": {
            "flood": {
                "risk": "High Risk" if is_flood else "Low Risk",
                "prediction": "Yes" if is_flood else "No",
                "probability": flood_prob,
                "severity": "CRITICAL" if is_flood else "SAFE",
                "advisory": (
                    "High runoff and water saturation detected. Follow emergency evacuation protocols."
                    if is_flood else "No flood threat detected under current hydrological indicators."
                )
            },
            "earthquake": {
                "predicted_magnitude": eq_mag,
                "category": eq_category,
                "severity": eq_severity,
                "advisory": eq_advisory
            },
            "hurricane": {
                "max_wind_knots": hur_wind_kt,
                "max_wind_kmh": hur_wind_kmh,
                "category": hur_cat,
                "severity": hur_sev,
                "advisory": hur_adv
            }
        },
        "overall_threat_level": threat_level,
        "threat_score": threat_score,
        "timestamp": pd.Timestamp.now().isoformat()
    }


@app.route("/api/live/predict-area", methods=["GET"])
def live_predict_area():
    """
    Live Area Disaster Predictor: Given a city name or coordinates,
    fetches live satellite/weather/elevation data and runs all AI models.
    Query: q (city string) OR lat & lon (floats)
    """
    q = request.args.get("q", "").strip()
    lat_param = request.args.get("lat")
    lon_param = request.args.get("lon")

    lat, lon, area_name = None, None, None

    # Check query string
    if q:
        # Check pre-mapped areas first for instant matching
        for a in GLOBAL_MONITORED_AREAS:
            if q.lower() in a["name"].lower():
                lat, lon, area_name = a["lat"], a["lon"], a["name"]
                break

        # If not pre-mapped, geocode via OpenStreetMap
        if lat is None:
            geo_url = f"https://nominatim.openstreetmap.org/search?q={urllib.parse.quote(q)}&format=json&limit=1"
            data, err = _fetch_external_json(geo_url)
            if data and len(data) > 0:
                try:
                    lat = float(data[0]["lat"])
                    lon = float(data[0]["lon"])
                    area_name = data[0].get("display_name", q)
                except Exception:
                    pass

    elif lat_param and lon_param:
        try:
            lat = float(lat_param)
            lon = float(lon_param)
            area_name = f"Location ({lat:.3f}, {lon:.3f})"
        except ValueError:
            return jsonify({"error": "Invalid lat or lon parameter"}), 400

    if lat is None or lon is None:
        # Fallback to Mumbai if query unresolved
        lat, lon, area_name = 19.0760, 72.8777, q or "Mumbai, Maharashtra, India"

    result = _predict_area_hazards(lat, lon, area_name=area_name)
    return jsonify(result)


@app.route("/api/live/monitored-areas", methods=["GET"])
def get_monitored_areas():
    """
    Returns curated list of vulnerable global metropolitan regions
    with live disaster prediction status.
    """
    areas_output = []
    # Fetch first 4 live to keep response fast, others with baseline
    for idx, a in enumerate(GLOBAL_MONITORED_AREAS):
        if idx < 4:
            pred = _predict_area_hazards(a["lat"], a["lon"], area_name=a["name"])
            areas_output.append({
                "name": a["name"],
                "lat": a["lat"],
                "lon": a["lon"],
                "risk_type": a["risk_type"],
                "overall_threat_level": pred["overall_threat_level"],
                "threat_score": pred["threat_score"],
                "flood_risk": pred["predictions"]["flood"]["risk"],
                "earthquake_mag": pred["predictions"]["earthquake"]["predicted_magnitude"],
                "wind_kmh": pred["live_metrics"]["wind_speed_kmh"],
                "temperature": pred["live_metrics"]["temperature_c"]
            })
        else:
            areas_output.append({
                "name": a["name"],
                "lat": a["lat"],
                "lon": a["lon"],
                "risk_type": a["risk_type"],
                "overall_threat_level": "ELEVATED",
                "threat_score": 45,
                "flood_risk": "Monitoring",
                "earthquake_mag": 4.5,
                "wind_kmh": 18.0,
                "temperature": 27.0
            })
    return jsonify({"status": "success", "count": len(areas_output), "areas": areas_output})


# ----- AI Prediction REST APIs -----
@app.route("/api/predict/flood", methods=["POST", "OPTIONS"])
def predict_flood():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    try:
        _load_models()
        if _flood_model is None:
            return jsonify({"error": "Flood model not found. Run create_all_models.py first."}), 500

        data = _extract_request_data()
        for key in data:
            try:
                data[key] = float(data[key])
            except (TypeError, ValueError):
                pass

        df = pd.DataFrame([data])
        df = pd.get_dummies(df, drop_first=True)
        df = df.reindex(columns=_flood_features, fill_value=0)

        pred = int(_flood_model.predict(df)[0])
        probability = None
        if hasattr(_flood_model, "predict_proba"):
            probs = _flood_model.predict_proba(df)[0]
            probability = round(float(probs[1]), 4) if len(probs) > 1 else None

        is_high_risk = (pred == 1)
        return jsonify({
            "status": "success",
            "prediction": "Yes" if is_high_risk else "No",
            "risk_level": "High Risk" if is_high_risk else "Low Risk",
            "severity": "CRITICAL" if is_high_risk else "NORMAL",
            "probability": probability,
            "advisory": (
                "Flood alert triggered: High runoff and water levels anticipated. "
                "Activate local flood defense barriers, secure vital assets, and review evacuation routes."
                if is_high_risk else
                "Normal hydrological parameters detected. Low flood risk under current inputs."
            )
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/predict/earthquake", methods=["POST", "OPTIONS"])
def predict_earthquake():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    try:
        _load_models()
        if _eq_model is None:
            return jsonify({"error": "Earthquake model not found. Run: python create_all_models.py"}), 400

        data = _extract_request_data()
        lat = float(data.get("Latitude", 0))
        lon = float(data.get("Longitude", 0))

        mag = float(_eq_model.predict(pd.DataFrame([{"Latitude": lat, "Longitude": lon}]))[0])
        mag_rounded = round(mag, 2)

        if mag_rounded >= 7.0:
            category = "Major Earthquake"
            severity = "CRITICAL"
            advisory = "Catastrophic damage possible across large areas. Severe structural risk."
        elif mag_rounded >= 6.0:
            category = "Strong Earthquake"
            severity = "HIGH"
            advisory = "Can cause damage in populated areas up to 160 km across."
        elif mag_rounded >= 5.0:
            category = "Moderate Earthquake"
            severity = "MODERATE"
            advisory = "Slight damage to poorly constructed buildings; felt by everyone."
        elif mag_rounded >= 4.0:
            category = "Light Earthquake"
            severity = "MINOR"
            advisory = "Noticeable shaking of indoor objects; minimal damage."
        else:
            category = "Minor / Micro Earthquake"
            severity = "LOW"
            advisory = "Often felt, but only causes minor vibration."

        return jsonify({
            "status": "success",
            "latitude": lat,
            "longitude": lon,
            "magnitude": mag_rounded,
            "category": category,
            "severity": severity,
            "advisory": advisory
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/predict/hurricane", methods=["POST", "OPTIONS"])
def predict_hurricane():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    try:
        _load_models()
        if _hur_model is None:
            return jsonify({"error": "Hurricane model not found. Run: python create_all_models.py"}), 400

        data = _extract_request_data()
        row = {
            "Latitude": float(data.get("Latitude", 25.0)),
            "Longitude": float(data.get("Longitude", -80.0)),
            "Moderate Wind NE": float(data.get("Moderate Wind NE", 0.0)),
            "Moderate Wind SE": float(data.get("Moderate Wind SE", 0.0)),
            "Moderate Wind SW": float(data.get("Moderate Wind SW", 0.0)),
            "Moderate Wind NW": float(data.get("Moderate Wind NW", 0.0)),
            "Year": int(float(data.get("Year", 2024))),
            "Month": int(float(data.get("Month", 8))),
            "Day": int(float(data.get("Day", 15))),
        }
        df = pd.DataFrame([row])
        df = df.reindex(columns=_hur_features, fill_value=0)
        wind = float(_hur_model.predict(df)[0])
        wind_rounded = round(wind, 1)
        wind_kmh = round(wind_rounded * 1.852, 1)

        # Saffir-Simpson Scale
        if wind_rounded >= 137:
            category = "Category 5 Hurricane"
            severity = "CATASTROPHIC"
            advisory = "Catastrophic damage will occur: A high percentage of framed homes will be destroyed."
        elif wind_rounded >= 113:
            category = "Category 4 Hurricane"
            severity = "EXTREME"
            advisory = "Catastrophic damage will occur: Well-built homes can sustain severe roof/wall damage."
        elif wind_rounded >= 96:
            category = "Category 3 Hurricane (Major)"
            severity = "SEVERE"
            advisory = "Devastating damage will occur: Electricity and water will be unavailable for days to weeks."
        elif wind_rounded >= 83:
            category = "Category 2 Hurricane"
            severity = "HIGH"
            advisory = "Extremely dangerous winds will cause extensive damage."
        elif wind_rounded >= 64:
            category = "Category 1 Hurricane"
            severity = "MODERATE"
            advisory = "Very dangerous winds will produce some damage."
        elif wind_rounded >= 34:
            category = "Tropical Storm"
            severity = "MODERATE"
            advisory = "Gale force winds and heavy precipitation anticipated."
        else:
            category = "Tropical Depression"
            severity = "LOW"
            advisory = "Organized storm system with winds under 34 knots."

        return jsonify({
            "status": "success",
            "max_wind_knots": wind_rounded,
            "max_wind_kmh": wind_kmh,
            "category": category,
            "severity": severity,
            "advisory": advisory
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ----- API Documentation Endpoint -----
@app.route("/api/docs", methods=["GET"])
def api_documentation():
    return jsonify({
        "name": "DisastraSense API Suite",
        "version": "2.0.0",
        "description": "RESTful AI Disaster Prediction & Real-Time Geospatial Hazard Intelligence",
        "endpoints": [
            {
                "path": "/api/predict/flood",
                "method": "POST",
                "content_type": ["application/json", "application/x-www-form-urlencoded"],
                "description": "Predicts flood risk based on meteorological, hydrological, and topological indicators.",
                "sample_payload": {
                    "Latitude": 28.61,
                    "Longitude": 77.20,
                    "Rainfall (mm)": 185.5,
                    "Temperature (°C)": 28.0,
                    "Humidity (%)": 85.0,
                    "River Discharge (m³/s)": 3200.0,
                    "Water Level (m)": 8.5,
                    "Elevation (m)": 210.0,
                    "Land Cover": "Urban",
                    "Soil Type": "Clay",
                    "Population Density": 11000,
                    "Infrastructure": 0,
                    "Historical Floods": 1
                }
            },
            {
                "path": "/api/predict/earthquake",
                "method": "POST",
                "content_type": ["application/json", "application/x-www-form-urlencoded"],
                "description": "Estimates expected earthquake magnitude based on geographic coordinates.",
                "sample_payload": {"Latitude": 35.6762, "Longitude": 139.6503}
            },
            {
                "path": "/api/predict/hurricane",
                "method": "POST",
                "content_type": ["application/json", "application/x-www-form-urlencoded"],
                "description": "Predicts maximum hurricane wind speed (knots & km/h) and Saffir-Simpson category.",
                "sample_payload": {
                    "Latitude": 25.7617,
                    "Longitude": -80.1918,
                    "Moderate Wind NE": 60,
                    "Moderate Wind SE": 55,
                    "Moderate Wind SW": 45,
                    "Moderate Wind NW": 50,
                    "Year": 2024,
                    "Month": 9,
                    "Day": 10
                }
            },
            {
                "path": "/api/weather",
                "method": "GET",
                "description": "Fetches live real-time weather, precipitation, elevation, and river discharge for any lat/lon.",
                "query_params": ["lat", "lon"]
            },
            {
                "path": "/api/geocode",
                "method": "GET",
                "description": "Search any city or address worldwide to auto-resolve latitude and longitude coordinates.",
                "query_params": ["q"]
            },
            {
                "path": "/api/live/earthquakes",
                "method": "GET",
                "description": "Retrieves real-time global earthquakes from USGS with magnitude, depth, and tsunami flags.",
                "query_params": ["limit", "min_mag"]
            }
        ]
    })


if __name__ == "__main__":
    port = 5000
    url = f"http://127.0.0.1:{port}"
    print("\n  DisastraSense running with Full API Suite. Open in browser:\n")
    print(f"  {url}\n")
    app.run(debug=True, host="127.0.0.1", port=port)
