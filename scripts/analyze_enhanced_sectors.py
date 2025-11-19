"""
Enhanced Sector Analysis with Intermediate Timing
Combines sector intelligence with 6 timing points, wind analysis, top speed tracking
"""

import pandas as pd
import numpy as np
import os
import sys
import json

# Add ai_engine to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ai_engine.enhanced_data_loader import EnhancedDataLoader, IntermediateTimingAnalyzer, WindAnalyzer

print("=" * 80)
print("ENHANCED SECTOR ANALYSIS")
print("Intermediate Timing Points + Wind Analysis + Top Speed Tracking")
print("=" * 80)

# Create output directory
os.makedirs('outputs/enhanced_sector_analysis', exist_ok=True)

# Initialize loaders and analyzers
loader = EnhancedDataLoader(race_number=1)
timing_analyzer = IntermediateTimingAnalyzer()
wind_analyzer = WindAnalyzer()

# Load all data
print("\n[1/5] Loading enhanced data sources...")
data = loader.load_all_data()

sectors_df = data['sectors']
weather_df = data['weather']

if sectors_df.empty:
    print("[ERROR] No sector data available")
    exit(1)

# Merge weather with laps
print("\n[2/5] Merging weather data with laps...")
sectors_df = loader.merge_weather_with_laps(sectors_df, weather_df)

# Analyze each driver
print("\n[3/5] Analyzing drivers with intermediate timing...")

all_driver_analysis = []
all_timing_recommendations = []

# Get unique drivers - use NUMBER column (actual car number), not DRIVER_NUMBER
# Clean column names first
sectors_df.columns = sectors_df.columns.str.strip()

if 'NUMBER' in sectors_df.columns:
    drivers = sectors_df['NUMBER'].unique()
    driver_col = 'NUMBER'
elif 'DRIVER_NUMBER' in sectors_df.columns:
    drivers = sectors_df['DRIVER_NUMBER'].unique()
    driver_col = 'DRIVER_NUMBER'
else:
    print("[ERROR] No driver identifier column found")
    exit(1)

for driver_num in drivers:
    driver_laps = sectors_df[sectors_df[driver_col] == driver_num].copy()
    
    if len(driver_laps) == 0:
        continue
    
    driver_name = driver_laps['DRIVER_NAME'].iloc[0] if 'DRIVER_NAME' in driver_laps.columns else f"Driver {driver_num}"
    
    print(f"  Analyzing {driver_name}...")
    
    # Timing point analysis
    timing_analysis = timing_analyzer.analyze_driver_timing_points(driver_laps)
    
    # Wind analysis
    wind_analysis = wind_analyzer.analyze_wind_impact(driver_laps)
    
    # Top speed analysis
    if 'TOP_SPEED' in driver_laps.columns:
        top_speed_stats = {
            'avg_top_speed': float(driver_laps['TOP_SPEED'].mean()),
            'max_top_speed': float(driver_laps['TOP_SPEED'].max()),
            'min_top_speed': float(driver_laps['TOP_SPEED'].min()),
            'top_speed_variance': float(driver_laps['TOP_SPEED'].std())
        }
    else:
        top_speed_stats = {}
    
    # Combine analysis
    driver_analysis = {
        'driver_number': int(driver_num),
        'driver_name': driver_name,
        'total_laps': len(driver_laps),
        'timing_analysis': timing_analysis,
        'wind_analysis': wind_analysis,
        'top_speed_stats': top_speed_stats
    }
    
    all_driver_analysis.append(driver_analysis)
    
    # Generate timing-based recommendations
    timing_recs = timing_analyzer.generate_timing_point_recommendations(timing_analysis)
    
    for rec in timing_recs:
        rec['driver_number'] = int(driver_num)
        rec['driver_name'] = driver_name
        all_timing_recommendations.append(rec)

print(f"[OK] Analyzed {len(all_driver_analysis)} drivers")

# Save analysis
print("\n[4/5] Saving enhanced sector analysis...")

# Save full analysis as JSON
with open('outputs/enhanced_sector_analysis/full_analysis.json', 'w') as f:
    json.dump(all_driver_analysis, f, indent=2)

print("[OK] Saved: outputs/enhanced_sector_analysis/full_analysis.json")

# Save timing recommendations as CSV
if all_timing_recommendations:
    timing_recs_df = pd.DataFrame(all_timing_recommendations)
    timing_recs_df.to_csv('outputs/enhanced_sector_analysis/timing_recommendations.csv', index=False)
    print(f"[OK] Saved: outputs/enhanced_sector_analysis/timing_recommendations.csv ({len(timing_recs_df)} recommendations)")

# Generate summary report
print("\n[5/5] Generating summary report...")

with open('outputs/enhanced_sector_analysis/analysis_report.txt', 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("ENHANCED SECTOR ANALYSIS REPORT\n")
    f.write("Intermediate Timing Points + Wind Analysis + Top Speed\n")
    f.write("=" * 80 + "\n\n")
    
    f.write(f"Drivers Analyzed: {len(all_driver_analysis)}\n")
    f.write(f"Timing Recommendations Generated: {len(all_timing_recommendations)}\n")
    f.write(f"Total Laps Analyzed: {len(sectors_df)}\n\n")
    
    # Wind summary
    if not weather_df.empty:
        f.write("=" * 80 + "\n")
        f.write("WIND CONDITIONS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Average Wind Speed: {weather_df['WIND_SPEED'].mean():.2f} km/h\n")
        f.write(f"Max Wind Speed: {weather_df['WIND_SPEED'].max():.2f} km/h\n")
        f.write(f"Average Wind Direction: {weather_df['WIND_DIRECTION'].mean():.0f}°\n")
        f.write(f"Pressure: {weather_df['PRESSURE'].mean():.1f} mbar\n\n")
    
    # Top recommendations by driver
    f.write("=" * 80 + "\n")
    f.write("TOP TIMING POINT IMPROVEMENTS BY DRIVER\n")
    f.write("=" * 80 + "\n\n")
    
    for driver_analysis in all_driver_analysis[:10]:  # Top 10 drivers
        f.write(f"\n{driver_analysis['driver_name']}\n")
        f.write("-" * 40 + "\n")
        
        timing = driver_analysis['timing_analysis']
        
        if timing.get('weakest_segments'):
            f.write("\nWeakest Segments:\n")
            for seg in timing['weakest_segments'][:3]:
                f.write(f"  • {seg['location']}: {seg['potential_gain']:.3f}s potential gain\n")
                f.write(f"    Type: {seg['type']}, Avg: {seg['avg_time']:.3f}s, Best: {seg['best_time']:.3f}s\n")
        
        # Top speed
        if driver_analysis['top_speed_stats']:
            ts = driver_analysis['top_speed_stats']
            f.write(f"\nTop Speed: {ts['avg_top_speed']:.1f} km/h avg (max: {ts['max_top_speed']:.1f} km/h)\n")
        
        # Wind impact
        if driver_analysis['wind_analysis']:
            wind = driver_analysis['wind_analysis']
            f.write(f"\nWind Impact: {wind.get('severity', 'Unknown')} ({wind.get('avg_wind_speed', 0):.1f} km/h)\n")
        
        f.write("\n")

print("[OK] Saved: outputs/enhanced_sector_analysis/analysis_report.txt")

print("\n" + "=" * 80)
print("ENHANCED SECTOR ANALYSIS COMPLETE")
print("=" * 80)
print(f"\nResults:")
print(f"  ✓ {len(all_driver_analysis)} drivers analyzed")
print(f"  ✓ {len(all_timing_recommendations)} timing-based recommendations")
print(f"  ✓ 6 timing points per lap analyzed")
print(f"  ✓ Wind impact calculated")
print(f"  ✓ Top speed tracking enabled")
print(f"\nOutputs saved to: outputs/enhanced_sector_analysis/")
