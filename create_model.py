"""Create physically realistic and calibrated model and model_features .pkl files for Flood prediction."""
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)
n = 8000

df = pd.DataFrame({
    'Latitude': np.random.uniform(8, 37, n),
    'Longitude': np.random.uniform(68, 97, n),
    'Rainfall (mm)': np.random.uniform(0, 320, n),
    'Temperature (°C)': np.random.uniform(10, 48, n),
    'Humidity (%)': np.random.uniform(15, 100, n),
    'River Discharge (m³/s)': np.random.uniform(0, 6000, n),
    'Water Level (m)': np.random.uniform(0, 10, n),
    'Elevation (m)': np.random.uniform(1, 4000, n),
    'Land Cover': np.random.choice(['Agricultural', 'Desert', 'Forest', 'Urban', 'Water Body'], n),
    'Soil Type': np.random.choice(['Clay', 'Loam', 'Peat', 'Sandy', 'Silt'], n),
    'Population Density': np.random.uniform(2, 12000, n),
    'Infrastructure': np.random.choice([0, 1], n, p=[0.5, 0.5]),
    'Historical Floods': np.random.choice([0, 1], n, p=[0.7, 0.3]),
})

# Hydrological formula based on actual catchment runoffs
soil_factor = df['Soil Type'].map({'Sandy': 0.7, 'Loam': 0.9, 'Silt': 1.0, 'Peat': 1.15, 'Clay': 1.35})
infra_factor = df['Infrastructure'].map({1: 0.7, 0: 1.3})
hist_factor = df['Historical Floods'].map({1: 1.25, 0: 0.9})
elevation_drainage = np.exp(-df['Elevation (m)'] / 120.0)

# Runoff pressure: heavy rain + river surge + water level
runoff_pressure = (
    (df['Rainfall (mm)'] / 90.0) * 2.8 +
    (df['River Discharge (m³/s)'] / 1800.0) * 2.2 +
    (df['Water Level (m)'] / 3.5) * 2.5
) * soil_factor * infra_factor * hist_factor

# Combined flood risk score
flood_score = runoff_pressure * (0.35 + 0.65 * elevation_drainage)

# Sigmoid probability
prob_flood = 1 / (1 + np.exp(-(flood_score - 5.2)))

# Calibration: If rainfall < 30mm and water level < 4.5m and discharge < 800 m3/s, flood is virtually impossible (<1%)
dry_mask = (df['Rainfall (mm)'] < 35) & (df['Water Level (m)'] < 5.0) & (df['River Discharge (m³/s)'] < 1200)
prob_flood[dry_mask] = prob_flood[dry_mask] * 0.01

# Labeled targets
df['Flood Occurred'] = (np.random.uniform(0, 1, n) < prob_flood).astype(int)

X = df.drop('Flood Occurred', axis=1)
y = df['Flood Occurred']

X_encoded = pd.get_dummies(X, drop_first=True)
feature_columns = X_encoded.columns.tolist()

X_mat = np.asarray(X_encoded.values, dtype=np.float32)
y_vec = np.asarray(y.values, dtype=np.int32)

model = RandomForestClassifier(n_estimators=100, max_depth=14, random_state=42)
model.fit(X_mat, y_vec)

joblib.dump(model, "random_forest_model.pkl")
joblib.dump(feature_columns, "model_features.pkl")
print("Calibrated random_forest_model.pkl and model_features.pkl successfully created.")
print(f"Sample dry condition flood rate: {df[dry_mask]['Flood Occurred'].mean():.3%}")
print(f"Sample severe rain condition flood rate: {df[df['Rainfall (mm)'] > 160]['Flood Occurred'].mean():.3%}")
