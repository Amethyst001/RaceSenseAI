"""
Official Results & Best Laps Analysis
Analyze official race results and best 10 laps to validate predictions
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("PHASE 9: OFFICIAL RESULTS & BEST LAPS ANALYSIS")
print("=" * 80)

# ============================================================================
# SECTION 1: Load Official Results
# ============================================================================

print("\n[1/4] Loading official race results...")
official = pd.read_csv('indianapolis/indianapolis/03_GR Cup Race 1 Official Results.CSV', 
                       sep=';', encoding='utf-8')
official.columns = official.columns.str.strip()

print(f"[OK] Loaded {len(official)} official results")
print(f"[INFO] Columns: {official.columns.tolist()[:10]}...")

# ============================================================================
# SECTION 2: Load Best 10 Laps
# ============================================================================

print("\n[2/4] Loading best 10 laps per driver...")
best_laps = pd.read_csv('indianapolis/indianapolis/99_Best 10 Laps By Driver_Race 1.CSV',
                        sep=';', encoding='utf-8')
best_laps.columns = best_laps.columns.str.strip()

print(f"[OK] Loaded {len(best_laps)} best lap records")
print(f"[INFO] Drivers with best laps: {best_laps['NUMBER'].nunique()}")

# ============================================================================
# SECTION 3: Analyze Best Laps Consistency
# ============================================================================

print("\n[3/4] Analyzing lap consistency from best 10 laps...")

# Parse lap time format (e.g., "1:40.123")
def parse_lap_time(time_str):
    try:
        if pd.isna(time_str) or time_str == '':
            return np.nan
        parts = str(time_str).split(':')
        if len(parts) == 2:
            minutes = float(parts[0])
            seconds = float(parts[1])
            return minutes * 60 + seconds
        else:
            return float(time_str)
    except:
        return np.nan

# Analyze consistency for each driver
consistency_analysis = []

for driver_num in best_laps['NUMBER'].unique():
    driver_row = best_laps[best_laps['NUMBER'] == driver_num].iloc[0]
    
    # Extract all best lap times
    lap_times = []
    for i in range(1, 11):
        col_name = f'BESTLAP_{i}'
        if col_name in best_laps.columns:
            lap_time = parse_lap_time(driver_row[col_name])
            if not pd.isna(lap_time):
                lap_times.append(lap_time)
    
    if len(lap_times) < 3:
        continue
    
    lap_times = np.array(lap_times)
    
    analysis = {
        'driver_number': driver_num,
        'driver_name': f"GR86 #{int(driver_num)}",
        'best_lap': lap_times.min(),
        'worst_of_best_10': lap_times.max(),
        'avg_of_best_10': lap_times.mean(),
        'std_of_best_10': lap_times.std(),
        'range': lap_times.max() - lap_times.min(),
        'consistency_score': (1 - (lap_times.std() / lap_times.mean())) * 100,
        'num_laps': len(lap_times)
    }
    
    # Identify if driver is improving or degrading
    if len(lap_times) >= 5:
        first_half = lap_times[:len(lap_times)//2].mean()
        second_half = lap_times[len(lap_times)//2:].mean()
        analysis['trend'] = 'Improving' if second_half < first_half else 'Degrading'
        analysis['trend_magnitude'] = abs(second_half - first_half)
    else:
        analysis['trend'] = 'Insufficient data'
        analysis['trend_magnitude'] = 0
    
    consistency_analysis.append(analysis)

consistency_df = pd.DataFrame(consistency_analysis)
consistency_df = consistency_df.sort_values('best_lap')

print(f"[OK] Analyzed consistency for {len(consistency_df)} drivers")
print(f"\nTop 5 most consistent drivers (lowest std):")
print(consistency_df.nsmallest(5, 'std_of_best_10')[['driver_name', 'best_lap', 'std_of_best_10', 'consistency_score']])

# ============================================================================
# SECTION 4: Compare with Predictions
# ============================================================================

print("\n[4/4] Comparing official results with predictions...")

# Load prediction data
try:
    predictions = pd.read_csv('outputs/predictions/per_driver_predictions.csv')
    has_predictions = True
except:
    predictions = pd.DataFrame()
    has_predictions = False
    print("[WARN] No predictions found to compare")

if has_predictions:
    comparison = []
    
    for _, pred in predictions.iterrows():
        driver_id = pred['driver_id']
        driver_name = pred['driver_name']
        predicted_best = pred['best_lap']
        predicted_next = pred.get('predicted_next_lap', predicted_best)
        
        # Find in official best laps
        official_data = consistency_df[consistency_df['driver_number'] == driver_id]
        
        if len(official_data) > 0:
            official_best = official_data.iloc[0]['best_lap']
            official_avg = official_data.iloc[0]['avg_of_best_10']
            
            comparison.append({
                'driver_name': driver_name,
                'driver_id': driver_id,
                'predicted_best': predicted_best,
                'official_best': official_best,
                'prediction_error': abs(predicted_best - official_best),
                'error_percentage': abs(predicted_best - official_best) / official_best * 100,
                'predicted_next': predicted_next,
                'official_avg_best_10': official_avg,
                'consistency_score': official_data.iloc[0]['consistency_score']
            })
    
    comparison_df = pd.DataFrame(comparison)
    
    if len(comparison_df) > 0:
        print(f"\n[OK] Compared {len(comparison_df)} drivers")
        print(f"\nPrediction Accuracy:")
        print(f"  Mean Absolute Error: {comparison_df['prediction_error'].mean():.3f}s")
        print(f"  Median Error: {comparison_df['prediction_error'].median():.3f}s")
        print(f"  Max Error: {comparison_df['prediction_error'].max():.3f}s")
        print(f"  Mean Error %: {comparison_df['error_percentage'].mean():.2f}%")
        
        print(f"\nTop 5 most accurate predictions:")
        print(comparison_df.nsmallest(5, 'prediction_error')[['driver_name', 'predicted_best', 'official_best', 'prediction_error']])
        
        print(f"\nTop 5 least accurate predictions:")
        print(comparison_df.nlargest(5, 'prediction_error')[['driver_name', 'predicted_best', 'official_best', 'prediction_error']])

# ============================================================================
# SECTION 5: Official Race Results Analysis
# ============================================================================

print("\n[5/5] Analyzing official race results...")

# Get final positions and times
if 'POSITION' in official.columns and 'TOTAL_TIME' in official.columns:
    race_results = official[['POSITION', 'NUMBER', 'TOTAL_TIME', 'LAPS', 'FL_TIME']].copy()
    race_results = race_results.sort_values('POSITION')
    
    print(f"\n[OK] Race Results:")
    print(f"  Winner: Car #{race_results.iloc[0]['NUMBER']}")
    print(f"  Total Laps: {race_results.iloc[0]['LAPS']}")
    print(f"  Winning Time: {race_results.iloc[0]['TOTAL_TIME']}")
    
    # Calculate average lap time for each driver
    race_results['avg_lap_time'] = pd.to_numeric(race_results['TOTAL_TIME'], errors='coerce') / pd.to_numeric(race_results['LAPS'], errors='coerce')
    
    print(f"\nTop 5 finishers:")
    print(race_results.head(5)[['POSITION', 'NUMBER', 'LAPS', 'FL_TIME']])

# ============================================================================
# SECTION 6: Save Outputs
# ============================================================================

print("\n[6/6] Saving analysis outputs...")

import os
os.makedirs('outputs/official_results', exist_ok=True)

# Save consistency analysis
consistency_df.to_csv('outputs/official_results/best_laps_consistency.csv', index=False)
print("[OK] Saved: outputs/official_results/best_laps_consistency.csv")

# Save comparison
if has_predictions and len(comparison_df) > 0:
    comparison_df.to_csv('outputs/official_results/prediction_accuracy.csv', index=False)
    print("[OK] Saved: outputs/official_results/prediction_accuracy.csv")

# Save race results summary
if 'POSITION' in official.columns:
    race_summary = official[['POSITION', 'NUMBER', 'LAPS', 'TOTAL_TIME', 'FL_TIME', 'FL_LAPNUM']].copy()
    race_summary.to_csv('outputs/official_results/race_summary.csv', index=False)
    print("[OK] Saved: outputs/official_results/race_summary.csv")

# Generate summary report
with open('outputs/official_results/analysis_summary.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("OFFICIAL RESULTS & BEST LAPS ANALYSIS\n")
    f.write("=" * 80 + "\n\n")
    
    f.write("CONSISTENCY ANALYSIS (Best 10 Laps)\n")
    f.write("-" * 80 + "\n")
    f.write(f"Drivers analyzed: {len(consistency_df)}\n\n")
    
    f.write("Most Consistent Drivers (Top 5):\n")
    for idx, row in consistency_df.nsmallest(5, 'std_of_best_10').iterrows():
        f.write(f"  {row['driver_name']}: Best={row['best_lap']:.3f}s, Std={row['std_of_best_10']:.3f}s, Score={row['consistency_score']:.1f}%\n")
    
    f.write("\nFastest Drivers (Top 5):\n")
    for idx, row in consistency_df.nsmallest(5, 'best_lap').iterrows():
        f.write(f"  {row['driver_name']}: {row['best_lap']:.3f}s (Avg of best 10: {row['avg_of_best_10']:.3f}s)\n")
    
    if has_predictions and len(comparison_df) > 0:
        f.write("\n" + "=" * 80 + "\n")
        f.write("PREDICTION ACCURACY\n")
        f.write("-" * 80 + "\n")
        f.write(f"Mean Absolute Error: {comparison_df['prediction_error'].mean():.3f}s\n")
        f.write(f"Median Error: {comparison_df['prediction_error'].median():.3f}s\n")
        f.write(f"Mean Error Percentage: {comparison_df['error_percentage'].mean():.2f}%\n\n")
        
        f.write("Most Accurate Predictions (Top 5):\n")
        for idx, row in comparison_df.nsmallest(5, 'prediction_error').iterrows():
            f.write(f"  {row['driver_name']}: Predicted={row['predicted_best']:.3f}s, Actual={row['official_best']:.3f}s, Error={row['prediction_error']:.3f}s\n")

print("[OK] Saved: outputs/official_results/analysis_summary.txt")

print("\n" + "=" * 80)
print("PHASE 9 COMPLETE")
print("=" * 80)
print(f"\nKey Findings:")
print(f"  - {len(consistency_df)} drivers analyzed from best 10 laps")
if has_predictions and len(comparison_df) > 0:
    print(f"  - Prediction accuracy: {comparison_df['prediction_error'].mean():.3f}s MAE")
    print(f"  - Best prediction error: {comparison_df['prediction_error'].min():.3f}s")
    print(f"  - Worst prediction error: {comparison_df['prediction_error'].max():.3f}s")
print(f"\nOutputs saved to: outputs/official_results/")
