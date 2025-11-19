"""
Toyota GR Cup Indianapolis Race 1 - Multi-Model Prediction System
Complete implementation with conformal prediction intervals and per-driver model selection
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings
import os
import pickle
import json
from datetime import datetime

warnings.filterwarnings('ignore')

# Vehicle ID mapping (vehicle_number -> full vehicle_id)
VEHICLE_ID_MAP = {
    '0': 'GR86-002-000', '2': 'GR86-060-2', '3': 'GR86-040-3', '5': 'GR86-065-5',
    '7': 'GR86-006-7', '8': 'GR86-012-8', '11': 'GR86-035-11', '13': 'GR86-022-13',
    '16': 'GR86-010-16', '18': 'GR86-030-18', '21': 'GR86-047-21', '31': 'GR86-015-31',
    '46': 'GR86-033-46', '47': 'GR86-025-47', '55': 'GR86-016-55', '57': 'GR86-057-57',
    '72': 'GR86-026-72', '80': 'GR86-013-80', '86': 'GR86-021-86', '88': 'GR86-049-88',
    '89': 'GR86-028-89', '93': 'GR86-038-93', '98': 'GR86-036-98', '113': 'GR86-063-113',
}

# Import optional models
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("[WARNING] XGBoost not installed. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("[WARNING] LightGBM not installed. Install with: pip install lightgbm")

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("[WARNING] CatBoost not installed. Install with: pip install catboost")

# Setup output directories
os.makedirs('outputs/predictions', exist_ok=True)
os.makedirs('outputs/predictions/visualizations', exist_ok=True)
os.makedirs('models', exist_ok=True)

print("=" * 80)
print("TOYOTA GR CUP - MULTI-MODEL PREDICTION SYSTEM")
print("=" * 80)

# ============================================================================
# SECTION 1: Load and Prepare Data
# ============================================================================

print("\n[LOADING DATA] Importing processed telemetry and lap data...")

# Load merged telemetry data
df_merged = pd.read_csv('telemetry_analysis/merged_telemetry_data.csv')
df_merged['timestamp'] = pd.to_datetime(df_merged['timestamp'])

# Clean data: Remove invalid lap durations
initial_count = len(df_merged)
if 'lap_duration' in df_merged.columns:
    # Remove negative or zero durations
    df_merged = df_merged[df_merged['lap_duration'] > 0]
    # Remove unrealistic fast laps (< 60s)
    df_merged = df_merged[df_merged['lap_duration'] >= 60]
    # Remove unrealistic slow laps (> 200s)
    df_merged = df_merged[df_merged['lap_duration'] <= 200]
    removed = initial_count - len(df_merged)
    if removed > 0:
        print(f"[CLEANED] Removed {removed} invalid lap records")

# Load official results for driver names
df_results = pd.read_csv('indianapolis/indianapolis/03_GR Cup Race 1 Official Results.CSV', sep=';')
df_results.columns = df_results.columns.str.strip()

print(f"[OK] Loaded {len(df_merged)} merged records")

# Create driver name mapping
driver_names = {}
for _, row in df_results.iterrows():
    if pd.notna(row.get('NUMBER')):
        first = row.get('DRIVER_FIRSTNAME', '')
        last = row.get('DRIVER_SECONDNAME', '')
        name = f"{first} {last}".strip() if first or last else f"GR86 #{row['NUMBER']}"
        driver_names[str(row['NUMBER'])] = name

# ============================================================================
# SECTION 2: Feature Engineering
# ============================================================================

print("\n[PREPROCESSING] Preparing features for multi-model training...")

# Clean and prepare data
df_clean = df_merged.copy()
df_clean = df_clean.dropna(subset=['lap_duration'])
df_clean = df_clean[df_clean['lap_duration'] < 200]  # Remove outliers

# Create features
feature_columns = ['Speed', 'Engine_RPM', 'Throttle', 'Brake_Front', 'Brake_Rear',
                  'Accel_Lateral', 'Accel_Longitudinal', 'AIR_TEMP', 'TRACK_TEMP', 'HUMIDITY']

df_ml = df_clean.dropna(subset=feature_columns + ['lap_duration']).copy()

# Add derived features
df_ml['brake_total'] = df_ml['Brake_Front'].fillna(0) + df_ml['Brake_Rear'].fillna(0)
df_ml['g_force_total'] = np.sqrt(df_ml['Accel_Lateral']**2 + df_ml['Accel_Longitudinal']**2)
df_ml['lap_number'] = df_ml['lap']

# Weather-derived features
df_ml['temp_delta'] = df_ml['TRACK_TEMP'] - df_ml['AIR_TEMP']  # Track-air temp difference
df_ml['grip_index'] = (df_ml['TRACK_TEMP'] * 0.7 + (100 - df_ml['HUMIDITY']) * 0.3) / 100  # Estimated grip
df_ml['weather_score'] = (
    (df_ml['TRACK_TEMP'] - df_ml['TRACK_TEMP'].min()) / (df_ml['TRACK_TEMP'].max() - df_ml['TRACK_TEMP'].min()) * 0.5 +
    (1 - (df_ml['HUMIDITY'] - df_ml['HUMIDITY'].min()) / (df_ml['HUMIDITY'].max() - df_ml['HUMIDITY'].min())) * 0.5
)  # 0-1 score: higher = better conditions

# Sort and create lag features
df_ml = df_ml.sort_values(['vehicle_number', 'lap'])
df_ml['prev_lap_time'] = df_ml.groupby('vehicle_number')['lap_duration'].shift(1)
df_ml['avg_last_3_laps'] = df_ml.groupby('vehicle_number')['lap_duration'].transform(
    lambda x: x.rolling(window=3, min_periods=1).mean()
)

# Speed zones
df_ml['speed_zone'] = pd.cut(df_ml['Speed'], bins=[0, 60, 100, 200], labels=[1, 2, 3])
df_ml['speed_zone'] = df_ml['speed_zone'].astype(float)

# Final feature list (including weather-derived features)
ml_features = ['Speed', 'Engine_RPM', 'Throttle', 'brake_total', 'g_force_total',
               'AIR_TEMP', 'TRACK_TEMP', 'HUMIDITY', 'temp_delta', 'grip_index', 'weather_score',
               'lap_number', 'prev_lap_time', 'avg_last_3_laps', 'speed_zone']

df_ml = df_ml.dropna(subset=ml_features)

print(f"[OK] Prepared {len(df_ml)} samples with {len(ml_features)} features")

# Prepare data
X = df_ml[ml_features].values
y = df_ml['lap_duration'].values
driver_ids = df_ml['vehicle_number'].values

# Split: 60% train, 20% calibration, 20% test
X_temp, X_test, y_temp, y_test, driver_temp, driver_test = train_test_split(
    X, y, driver_ids, test_size=0.2, random_state=42
)
X_train, X_cal, y_train, y_cal, driver_train, driver_cal = train_test_split(
    X_temp, y_temp, driver_temp, test_size=0.25, random_state=42  # 0.25 of 80% = 20% overall
)

print(f"[OK] Split: Train={len(X_train)}, Calibration={len(X_cal)}, Test={len(X_test)}")

# ============================================================================
# SECTION 3: Train Multiple Models
# ============================================================================

print("\n[TRAINING] Building candidate model pool...")

models = {}
model_names = []

# 1. Random Forest
print("  [1/6] Training Random Forest...")
rf_model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
models['RandomForest'] = rf_model
model_names.append('RandomForest')

# 2. Gradient Boosting
print("  [2/6] Training Gradient Boosting...")
gb_model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
gb_model.fit(X_train, y_train)
models['GradientBoosting'] = gb_model
model_names.append('GradientBoosting')

# 3. Ridge Regression (baseline)
print("  [3/6] Training Ridge Regression...")
ridge_model = Ridge(alpha=1.0, random_state=42)
ridge_model.fit(X_train, y_train)
models['Ridge'] = ridge_model
model_names.append('Ridge')

# 4. XGBoost (if available)
if HAS_XGBOOST:
    print("  [4/6] Training XGBoost...")
    xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    xgb_model.fit(X_train, y_train)
    models['XGBoost'] = xgb_model
    model_names.append('XGBoost')
else:
    print("  [4/6] Skipping XGBoost (not installed)")

# 5. LightGBM (if available)
if HAS_LIGHTGBM:
    print("  [5/6] Training LightGBM...")
    lgb_model = lgb.LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, verbose=-1)
    lgb_model.fit(X_train, y_train)
    models['LightGBM'] = lgb_model
    model_names.append('LightGBM')
else:
    print("  [5/6] Skipping LightGBM (not installed)")

# 6. CatBoost (if available)
if HAS_CATBOOST:
    print("  [6/6] Training CatBoost...")
    cat_model = CatBoostRegressor(iterations=100, depth=5, learning_rate=0.1, random_state=42, verbose=0)
    cat_model.fit(X_train, y_train)
    models['CatBoost'] = cat_model
    model_names.append('CatBoost')
else:
    print("  [6/6] Skipping CatBoost (not installed)")

print(f"\n[OK] Trained {len(models)} models successfully")

# Save all models
for name, model in models.items():
    model_path = f'models/{name}.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"[OK] Saved: {model_path}")

# ============================================================================
# SECTION 4: Conformal Prediction Intervals
# ============================================================================

print("\n[CONFORMAL] Computing split-conformal prediction intervals...")

calibration_stats = {}
alpha = 0.1  # 90% confidence interval

for name, model in models.items():
    print(f"  Processing {name}...")
    
    # Predict on calibration set
    y_cal_pred = model.predict(X_cal)
    
    # Compute absolute residuals
    residuals = np.abs(y_cal - y_cal_pred)
    
    # Compute quantile
    n = len(residuals)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_residual = np.quantile(residuals, q_level)
    
    calibration_stats[name] = {
        'q_residual': float(q_residual),
        'alpha': alpha,
        'n_calibration': int(n),
        'mean_residual': float(np.mean(residuals)),
        'std_residual': float(np.std(residuals))
    }
    
    print(f"    Quantile residual (90%): {q_residual:.3f}s")

# Save calibration stats
with open('models/calibration_stats.json', 'w') as f:
    json.dump(calibration_stats, f, indent=2)
print("\n[OK] Saved: models/calibration_stats.json")

# ============================================================================
# SECTION 5: Per-Driver Model Evaluation & Selection
# ============================================================================

print("\n[EVALUATION] Evaluating models per driver...")

# Get unique drivers in test set
unique_drivers = np.unique(driver_test)
per_driver_results = []

for driver_id in unique_drivers:
    driver_name = VEHICLE_ID_MAP.get(str(driver_id), f"GR86-Unknown-{driver_id}")
    
    # Get driver's test samples
    driver_mask = driver_test == driver_id
    X_driver = X_test[driver_mask]
    y_driver = y_test[driver_mask]
    
    if len(y_driver) < 3:  # Need minimum samples
        continue
    
    driver_model_scores = []
    
    for name, model in models.items():
        # Predict
        y_pred = model.predict(X_driver)
        
        # Get conformal interval
        q_res = calibration_stats[name]['q_residual']
        y_lower = y_pred - q_res
        y_upper = y_pred + q_res
        
        # Compute metrics
        mae = mean_absolute_error(y_driver, y_pred)
        rmse = np.sqrt(mean_squared_error(y_driver, y_pred))
        
        # Interval coverage
        coverage = np.mean((y_driver >= y_lower) & (y_driver <= y_upper))
        
        # Average interval width
        interval_width = np.mean(y_upper - y_lower)
        
        # Calibration penalty
        calibration_penalty = abs(coverage - 0.90)
        
        # Normalize metrics for composite score
        # We'll normalize after collecting all scores
        driver_model_scores.append({
            'driver_id': driver_id,
            'driver_name': driver_name,
            'model': name,
            'mae': mae,
            'rmse': rmse,
            'coverage': coverage,
            'interval_width': interval_width,
            'calibration_penalty': calibration_penalty
        })
    
    per_driver_results.extend(driver_model_scores)

df_eval = pd.DataFrame(per_driver_results)

# Normalize metrics for composite score
df_eval['normalized_mae'] = (df_eval['mae'] - df_eval['mae'].min()) / (df_eval['mae'].max() - df_eval['mae'].min() + 1e-10)
df_eval['normalized_interval_width'] = (df_eval['interval_width'] - df_eval['interval_width'].min()) / (df_eval['interval_width'].max() - df_eval['interval_width'].min() + 1e-10)

# Composite score
df_eval['composite'] = (
    df_eval['normalized_mae'] +
    2 * df_eval['calibration_penalty'] +
    0.5 * df_eval['normalized_interval_width']
)

# Select best model per driver
df_selection = df_eval.loc[df_eval.groupby('driver_id')['composite'].idxmin()].copy()
df_selection = df_selection[['driver_id', 'driver_name', 'model', 'mae', 'coverage', 'interval_width', 'composite']]

# Save selection
df_selection.to_csv('models/per_driver_model_selection.csv', index=False)
print(f"[OK] Saved: models/per_driver_model_selection.csv")
print(f"[OK] Selected best models for {len(df_selection)} drivers")

# Print summary
print("\n  Model Selection Summary:")
model_counts = df_selection['model'].value_counts()
for model, count in model_counts.items():
    print(f"    {model}: {count} drivers")

# ============================================================================
# SECTION 6: Generate Predictions with Best Models
# ============================================================================

print("\n[PREDICTIONS] Generating predictions using per-driver best models...")

predictions_list = []

for _, selection_row in df_selection.iterrows():
    driver_id = selection_row['driver_id']
    driver_name = selection_row['driver_name']
    best_model_name = selection_row['model']
    
    # Get driver data
    driver_data = df_ml[df_ml['vehicle_number'] == driver_id].copy()
    
    if len(driver_data) < 5:
        continue
    
    # Get last lap features
    last_lap = driver_data.iloc[-1]
    features_for_pred = last_lap[ml_features].values.reshape(1, -1)
    
    # Load best model and predict
    best_model = models[best_model_name]
    predicted_time = best_model.predict(features_for_pred)[0]
    
    # Get ALL model predictions for agreement calculation
    all_model_predictions = []
    for model_name, model in models.items():
        pred = model.predict(features_for_pred)[0]
        all_model_predictions.append(pred)
    
    # Calculate model agreement (how many models agree within 10% of median)
    median_pred = np.median(all_model_predictions)
    threshold = median_pred * 0.10  # 10% threshold (more reasonable for lap times)
    models_in_agreement = sum(1 for pred in all_model_predictions if abs(pred - median_pred) <= threshold)
    model_agreement_score = (models_in_agreement / len(all_model_predictions)) * 100
    
    # Get conformal interval
    q_res = calibration_stats[best_model_name]['q_residual']
    prediction_lower = predicted_time - q_res
    prediction_upper = predicted_time + q_res
    
    # Calculate stats
    best_lap = driver_data['lap_duration'].min()
    avg_lap = driver_data['lap_duration'].mean()
    improvement_vs_best = predicted_time - best_lap
    
    # Confidence score (0-100)
    confidence_score = (1 - selection_row['composite']) * 100
    confidence_score = max(0, min(100, confidence_score))  # Clamp to 0-100
    
    predictions_list.append({
        'driver_id': driver_id,
        'driver_name': driver_name,
        'selected_model': best_model_name,
        'predicted_next_lap': predicted_time,
        'prediction_lower': prediction_lower,
        'prediction_upper': prediction_upper,
        'confidence_interval': q_res,
        'best_lap': best_lap,
        'avg_lap': avg_lap,
        'improvement_vs_best': improvement_vs_best,
        'mae': selection_row['mae'],
        'coverage': selection_row['coverage'],
        'interval_width': selection_row['interval_width'],
        'composite_score': selection_row['composite'],
        'confidence_score': confidence_score,
        'confidence': 'High' if confidence_score > 70 else 'Medium' if confidence_score > 40 else 'Low',
        'model_agreement_score': model_agreement_score,
        'models_in_agreement': models_in_agreement
    })

df_predictions = pd.DataFrame(predictions_list)
df_predictions = df_predictions.sort_values('predicted_next_lap')

# Save predictions
df_predictions.to_csv('outputs/predictions/per_driver_predictions.csv', index=False)
print(f"[OK] Saved: outputs/predictions/per_driver_predictions.csv")
print(f"[OK] Generated predictions for {len(df_predictions)} drivers")

# ============================================================================
# SECTION 7: Ensemble Mode (Top-3 Models)
# ============================================================================

print("\n[ENSEMBLE] Computing Top-3 ensemble predictions...")

ensemble_predictions = []

for driver_id in unique_drivers:
    driver_name = VEHICLE_ID_MAP.get(str(driver_id), f"GR86-Unknown-{driver_id}")
    
    # Get driver data
    driver_data = df_ml[df_ml['vehicle_number'] == driver_id].copy()
    
    if len(driver_data) < 5:
        continue
    
    # Get last lap features
    last_lap = driver_data.iloc[-1]
    features_for_pred = last_lap[ml_features].values.reshape(1, -1)
    
    # Get top-3 models for this driver
    driver_eval = df_eval[df_eval['driver_id'] == driver_id].sort_values('composite').head(3)
    
    if len(driver_eval) == 0:
        continue
    
    # Collect predictions from top-3
    top3_preds = []
    top3_lowers = []
    top3_uppers = []
    
    for _, model_row in driver_eval.iterrows():
        model_name = model_row['model']
        model = models[model_name]
        
        pred = model.predict(features_for_pred)[0]
        q_res = calibration_stats[model_name]['q_residual']
        
        top3_preds.append(pred)
        top3_lowers.append(pred - q_res)
        top3_uppers.append(pred + q_res)
    
    # Ensemble: average predictions, median bounds
    ensemble_pred = np.mean(top3_preds)
    ensemble_lower = np.median(top3_lowers)
    ensemble_upper = np.median(top3_uppers)
    
    # Stats
    best_lap = driver_data['lap_duration'].min()
    avg_lap = driver_data['lap_duration'].mean()
    
    ensemble_predictions.append({
        'driver_id': driver_id,
        'driver_name': driver_name,
        'ensemble_prediction': ensemble_pred,
        'ensemble_lower': ensemble_lower,
        'ensemble_upper': ensemble_upper,
        'ensemble_interval': ensemble_upper - ensemble_lower,
        'best_lap': best_lap,
        'avg_lap': avg_lap,
        'improvement_vs_best': ensemble_pred - best_lap,
        'top3_models': ', '.join(driver_eval['model'].tolist())
    })

df_ensemble = pd.DataFrame(ensemble_predictions)
df_ensemble = df_ensemble.sort_values('ensemble_prediction')

# Save ensemble predictions
df_ensemble.to_csv('outputs/predictions/ensemble_predictions.csv', index=False)
print(f"[OK] Saved: outputs/predictions/ensemble_predictions.csv")
print(f"[OK] Generated ensemble predictions for {len(df_ensemble)} drivers")

# ============================================================================
# SECTION 8: Generate Documentation
# ============================================================================

print("\n[DOCUMENTATION] Creating comprehensive README...")

readme_content = f"""# Multi-Model Prediction System

## Overview

This implements a complete multi-model prediction system with conformal prediction intervals and per-driver model selection for optimal accuracy.

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 🎯 System Architecture

### 1. Candidate Model Pool

Trained and evaluated {len(models)} models:
{chr(10).join([f"- **{name}**" for name in model_names])}

All models use the same cleaned feature matrix with {len(ml_features)} features.

### 2. Split-Conformal Prediction Intervals

- **Training Set**: {len(X_train)} samples (60%)
- **Calibration Set**: {len(X_cal)} samples (20%)
- **Test Set**: {len(X_test)} samples (20%)
- **Confidence Level**: 90% (alpha=0.1)

For each model:
- Compute absolute residuals on calibration set
- Calculate 90th percentile quantile
- Prediction interval: [pred - q_residual, pred + q_residual]

### 3. Per-Driver Model Selection

For each driver, evaluate all models using:
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **Coverage** (% of actual values within interval)
- **Interval Width** (average prediction interval size)
- **Calibration Penalty** (|coverage - 0.90|)

**Composite Score**:
```
composite = normalized_MAE + 2 * calibration_penalty + 0.5 * normalized_interval_width
```

Best model = lowest composite score

### 4. Ensemble Mode

For each driver:
- Select top-3 models by composite score
- Average their predictions
- Use median of lower/upper bounds for interval

---

## 📊 Model Performance Summary

### Model Selection Distribution:
{chr(10).join([f"- **{model}**: {count} drivers ({count/len(df_selection)*100:.1f}%)" for model, count in model_counts.items()])}

### Average Metrics Across All Drivers:
- **Mean MAE**: {df_selection['mae'].mean():.3f}s
- **Mean Coverage**: {df_selection['coverage'].mean():.1%}
- **Mean Interval Width**: {df_selection['interval_width'].mean():.3f}s
- **Mean Composite Score**: {df_selection['composite'].mean():.3f}

---

## 📁 Output Files

### 1. Model Files (`models/`)

#### Trained Models:
{chr(10).join([f"- `{name}.pkl` - Trained {name} model" for name in model_names])}

#### Calibration Data:
- `calibration_stats.json` - Conformal quantiles and statistics for each model
- `per_driver_model_selection.csv` - Best model selection per driver with metrics

### 2. Prediction Files (`outputs/predictions/`)

#### `per_driver_predictions.csv`
Complete predictions using per-driver best models:
- Driver information
- Selected model name
- Point prediction + confidence interval (lower, upper)
- Performance metrics (MAE, coverage, interval width)
- Composite score and confidence score (0-100)
- Improvement vs best/average lap

#### `ensemble_predictions.csv`
Top-3 ensemble predictions:
- Ensemble prediction (average of top-3)
- Ensemble interval (median bounds)
- List of top-3 models used
- Improvement metrics

---

## 🎮 How to Use

### For Race Strategy:
1. Load `per_driver_predictions.csv`
2. Check `selected_model` and `confidence_score` for reliability
3. Use `prediction_lower` and `prediction_upper` for risk assessment
4. Compare `improvement_vs_best` for potential gains

### For Driver Coaching:
1. Review selected model for each driver
2. Analyze why certain models work better (feature importance)
3. Use interval width as uncertainty indicator
4. Focus on drivers with high confidence scores

### For Technical Analysis:
1. Compare single-model vs ensemble predictions
2. Analyze model selection patterns
3. Review calibration statistics
4. Validate coverage rates

---

## 🔧 Technical Details

### Features Used ({len(ml_features)}):
{chr(10).join([f"- {feat}" for feat in ml_features])}

### Conformal Prediction:
- **Method**: Split conformal prediction
- **Quantile Level**: {(1-alpha)*100:.0f}%
- **Calibration Samples**: {len(X_cal)}
- **Validity**: Guaranteed coverage under exchangeability assumption

### Model Selection Criteria:
1. **Accuracy** (normalized MAE)
2. **Calibration** (2x penalty for coverage deviation)
3. **Precision** (0.5x penalty for wide intervals)

---

## 📈 Top Predictions

### Fastest Predicted (Best Model):
{chr(10).join([f"{i+1}. **{row['driver_name']}** ({row['selected_model']}): {row['predicted_next_lap']:.3f}s ± {row['confidence_interval']:.3f}s" for i, (_, row) in enumerate(df_predictions.head(5).iterrows())])}

### Highest Confidence Predictions:
{chr(10).join([f"{i+1}. **{row['driver_name']}**: {row['confidence_score']:.1f}% confidence ({row['selected_model']})" for i, (_, row) in enumerate(df_predictions.nlargest(5, 'confidence_score').iterrows())])}

### Best Ensemble Predictions:
{chr(10).join([f"{i+1}. **{row['driver_name']}**: {row['ensemble_prediction']:.3f}s [{row['ensemble_lower']:.3f}, {row['ensemble_upper']:.3f}]" for i, (_, row) in enumerate(df_ensemble.head(5).iterrows())])}

---

## 🚀 Next Steps

1. **Dashboard Integration**: Update dashboard with model selection widget
2. **Real-time Updates**: Implement live model switching based on performance
3. **Feature Analysis**: Deep dive into why certain models work for specific drivers
4. **Ensemble Tuning**: Experiment with weighted ensembles based on recent performance

---

## 📚 References

- Vovk, V., Gammerman, A., & Shafer, G. (2005). Algorithmic Learning in a Random World
- Lei, J., et al. (2018). Distribution-Free Predictive Inference For Regression
- Angelopoulos, A. N., & Bates, S. (2021). A Gentle Introduction to Conformal Prediction

---

*Generated by RaceSense AI - Multi-Model Prediction Engine*
"""

with open('outputs/predictions/README_predictions.md', 'w', encoding='utf-8') as f:
    f.write(readme_content)

print("[OK] Saved: README_predictions.md")

# ============================================================================
# SECTION: Weather Impact Analysis
# ============================================================================

print("\n" + "=" * 80)
print("WEATHER IMPACT ANALYSIS")
print("=" * 80)

# Analyze weather impact on lap times
weather_analysis = df_ml.groupby(pd.cut(df_ml['TRACK_TEMP'], bins=3, labels=['Cool', 'Medium', 'Warm'])).agg({
    'lap_duration': ['mean', 'std', 'count'],
    'AIR_TEMP': 'mean',
    'TRACK_TEMP': 'mean',
    'HUMIDITY': 'mean',
    'weather_score': 'mean'
}).round(3)

print("\n[WEATHER] Lap time by track temperature:")
print(weather_analysis)

# Calculate weather impact on predictions
weather_impact = {
    'avg_air_temp': float(df_ml['AIR_TEMP'].mean()),
    'avg_track_temp': float(df_ml['TRACK_TEMP'].mean()),
    'avg_humidity': float(df_ml['HUMIDITY'].mean()),
    'temp_range': {
        'min': float(df_ml['TRACK_TEMP'].min()),
        'max': float(df_ml['TRACK_TEMP'].max())
    },
    'optimal_conditions': {
        'track_temp': float(df_ml.loc[df_ml['lap_duration'].idxmin(), 'TRACK_TEMP']),
        'humidity': float(df_ml.loc[df_ml['lap_duration'].idxmin(), 'HUMIDITY']),
        'weather_score': float(df_ml.loc[df_ml['lap_duration'].idxmin(), 'weather_score'])
    },
    'weather_correlation': {
        'track_temp': float(df_ml[['lap_duration', 'TRACK_TEMP']].corr().iloc[0, 1]),
        'humidity': float(df_ml[['lap_duration', 'HUMIDITY']].corr().iloc[0, 1]),
        'weather_score': float(df_ml[['lap_duration', 'weather_score']].corr().iloc[0, 1])
    }
}

# Save weather analysis
with open('outputs/predictions/weather_impact.json', 'w') as f:
    json.dump(weather_impact, f, indent=2)

print("\n[OK] Weather impact analysis saved to 'outputs/predictions/weather_impact.json'")
print(f"\n[INSIGHT] Weather correlations with lap time:")
print(f"  Track Temperature: {weather_impact['weather_correlation']['track_temp']:.3f}")
print(f"  Humidity: {weather_impact['weather_correlation']['humidity']:.3f}")
print(f"  Weather Score: {weather_impact['weather_correlation']['weather_score']:.3f}")

print("\n" + "=" * 80)
print("MULTI-MODEL PREDICTION SYSTEM COMPLETE")
print("=" * 80)
print(f"\n✅ Trained {len(models)} models with {len(ml_features)} features (including 3 weather-derived)")
print(f"✅ Computed conformal intervals for all models")
print(f"✅ Selected best models for {len(df_selection)} drivers")
print(f"✅ Generated predictions with confidence intervals")
print(f"✅ Created ensemble predictions")
print(f"✅ Analyzed weather impact on performance")
print(f"✅ Saved all outputs and documentation")
print("\nReady for dashboard integration!")
