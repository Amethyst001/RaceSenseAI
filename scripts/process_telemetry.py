"""
Toyota GR Cup Indianapolis Race 1 - Telemetry Analysis
Integrates telemetry data with lap times and weather data
Generates AI-driven insights and performance recommendations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

# Create output directory for plots
os.makedirs('telemetry_analysis', exist_ok=True)

print("=" * 80)
print("TOYOTA GR CUP INDIANAPOLIS RACE 1 - TELEMETRY ANALYSIS")
print("=" * 80)

# ============================================================================
# SECTION 1: Load Race Data (Lap Times, Weather, Results)
# ============================================================================

print("\n[LAP DATA] Loading and calculating lap durations...")

# Load lap time, start, and end data
LAP_TIME_FILE = "indianapolis/indianapolis/R1_indianapolis_motor_speedway_lap_time.csv"
LAP_START_FILE = "indianapolis/indianapolis/R1_indianapolis_motor_speedway_lap_start.csv"
LAP_END_FILE = "indianapolis/indianapolis/R1_indianapolis_motor_speedway_lap_end.csv"

df_laps = pd.read_csv(LAP_TIME_FILE)
df_lap_start = pd.read_csv(LAP_START_FILE)
df_lap_end = pd.read_csv(LAP_END_FILE)

# Convert timestamps
df_laps['timestamp'] = pd.to_datetime(df_laps['timestamp'])
df_lap_start['timestamp'] = pd.to_datetime(df_lap_start['timestamp'])
df_lap_end['timestamp'] = pd.to_datetime(df_lap_end['timestamp'])

# Remove duplicates
df_lap_start = df_lap_start.drop_duplicates(subset=['vehicle_number', 'lap'], keep='first')
df_lap_end = df_lap_end.drop_duplicates(subset=['vehicle_number', 'lap'], keep='first')
df_laps = df_laps.drop_duplicates(subset=['vehicle_number', 'lap'], keep='first')

# Sort and merge
df_laps = df_laps.sort_values(['vehicle_number', 'lap'])

# Merge start and end times
df_laps = df_laps.merge(
    df_lap_start[['vehicle_number', 'lap', 'timestamp']],
    on=['vehicle_number', 'lap'],
    how='left',
    suffixes=('', '_start')
)
df_laps = df_laps.merge(
    df_lap_end[['vehicle_number', 'lap', 'timestamp']],
    on=['vehicle_number', 'lap'],
    how='left',
    suffixes=('', '_end')
)

# Calculate CORRECT lap duration (end - start)
df_laps['lap_duration'] = (
    df_laps['timestamp_end'] - df_laps['timestamp_start']
).dt.total_seconds()

# Clean data
df_laps = df_laps.dropna(subset=['timestamp_start', 'timestamp_end'])
df_laps = df_laps[df_laps['lap_duration'] > 0]  # Remove negative
df_laps = df_laps[df_laps['lap_duration'] >= 60]  # Remove too fast
df_laps_clean = df_laps[df_laps['lap_duration'] < 200].copy()  # Remove too slow

print(f"✓ Loaded and calculated {len(df_laps_clean)} valid lap records")

# Load weather data
WEATHER_FILE = "indianapolis/indianapolis/26_Weather_Race 1.CSV"
df_weather = pd.read_csv(WEATHER_FILE, sep=';')
df_weather['TIME_UTC_SECONDS'] = pd.to_numeric(df_weather['TIME_UTC_SECONDS'], errors='coerce')
print(f"✓ Loaded {len(df_weather)} weather records")

# Load official results
RESULTS_FILE = "indianapolis/indianapolis/03_GR Cup Race 1 Official Results.CSV"
df_results = pd.read_csv(RESULTS_FILE, sep=';')
df_results.columns = df_results.columns.str.strip()
print(f"✓ Loaded {len(df_results)} driver results")

# ============================================================================
# SECTION 2: Load Telemetry Data Efficiently (Chunked Processing)
# ============================================================================

print("\n[TELEMETRY DATA] Loading large telemetry file in chunks...")

TELEMETRY_FILE = "indianapolis/indianapolis/R1_indianapolis_motor_speedway_telemetry.csv"

# Define telemetry parameters we're interested in
TELEMETRY_PARAMS = {
    'speed': 'Speed',
    'nmot': 'Engine_RPM',
    'aps': 'Throttle',
    'pbrake_f': 'Brake_Front',
    'pbrake_r': 'Brake_Rear',
    'accx_can': 'Accel_Longitudinal',
    'accy_can': 'Accel_Lateral',
    'gear': 'Gear',
    'Steering_Angle': 'Steering_Angle'
}

# Process telemetry in chunks to handle large file
chunk_size = 100000
telemetry_chunks = []

print("Processing telemetry chunks...")
for i, chunk in enumerate(pd.read_csv(TELEMETRY_FILE, chunksize=chunk_size)):
    # Filter only relevant telemetry parameters
    chunk_filtered = chunk[chunk['telemetry_name'].isin(TELEMETRY_PARAMS.keys())].copy()
    telemetry_chunks.append(chunk_filtered)
    if (i + 1) % 10 == 0:
        print(f"  Processed {(i + 1) * chunk_size:,} rows...")

# Combine all chunks
df_telemetry = pd.concat(telemetry_chunks, ignore_index=True)
print(f"✓ Loaded {len(df_telemetry):,} telemetry records")

# Convert timestamp
df_telemetry['timestamp'] = pd.to_datetime(df_telemetry['timestamp'])
df_telemetry['telemetry_value'] = pd.to_numeric(df_telemetry['telemetry_value'], errors='coerce')

# ============================================================================
# SECTION 3: Pivot Telemetry Data & Calculate Per-Lap Statistics
# ============================================================================

print("\n[PROCESSING] Calculating per-lap telemetry statistics...")

# Group by vehicle, lap, and telemetry type to calculate statistics
telemetry_stats = df_telemetry.groupby(['vehicle_number', 'lap', 'telemetry_name']).agg({
    'telemetry_value': ['mean', 'max', 'min', 'std']
}).reset_index()

telemetry_stats.columns = ['vehicle_number', 'lap', 'telemetry_name', 'mean', 'max', 'min', 'std']

# Pivot to wide format for easier analysis
telemetry_wide = telemetry_stats.pivot_table(
    index=['vehicle_number', 'lap'],
    columns='telemetry_name',
    values='mean'
).reset_index()

# Rename columns for clarity
telemetry_wide.columns.name = None
telemetry_wide = telemetry_wide.rename(columns=TELEMETRY_PARAMS)

print(f"✓ Calculated statistics for {len(telemetry_wide)} vehicle-lap combinations")
print(f"✓ Telemetry parameters: {list(TELEMETRY_PARAMS.values())}")

# ============================================================================
# SECTION 4: Merge Telemetry with Lap Times and Weather
# ============================================================================

print("\n[MERGING] Combining telemetry with lap times and weather...")

# Merge telemetry with lap times
df_merged = df_laps_clean.merge(
    telemetry_wide,
    on=['vehicle_number', 'lap'],
    how='left'
)

# Add weather data (merge by nearest timestamp)
df_merged['lap_unix_time'] = df_merged['timestamp'].astype(np.int64) // 10**9
df_merged = pd.merge_asof(
    df_merged.sort_values('lap_unix_time'),
    df_weather[['TIME_UTC_SECONDS', 'AIR_TEMP', 'TRACK_TEMP', 'HUMIDITY']].sort_values('TIME_UTC_SECONDS'),
    left_on='lap_unix_time',
    right_on='TIME_UTC_SECONDS',
    direction='nearest',
    tolerance=60
)

print(f"✓ Merged dataset contains {len(df_merged)} records")
print(f"✓ Available columns: {df_merged.columns.tolist()}")

# ============================================================================
# SECTION 5: Pattern Analysis & Correlations
# ============================================================================

print("\n" + "=" * 80)
print("PATTERN ANALYSIS")
print("=" * 80)

# Remove rows with missing critical data
df_analysis = df_merged.dropna(subset=['lap_duration', 'Speed', 'Throttle']).copy()

# 1. Throttle/Brake patterns vs lap times
print("\n[1] Analyzing throttle/brake patterns vs lap times...")

if len(df_analysis) > 0:
    # Calculate correlations
    corr_throttle = df_analysis[['lap_duration', 'Throttle', 'Brake_Front', 'Speed']].corr()
    print("\nCorrelation Matrix (Throttle/Brake vs Lap Time):")
    print(corr_throttle['lap_duration'].sort_values(ascending=False))
    
    # Identify optimal throttle/brake balance
    df_analysis['brake_total'] = df_analysis['Brake_Front'].fillna(0) + df_analysis['Brake_Rear'].fillna(0)
    fast_laps = df_analysis[df_analysis['lap_duration'] < df_analysis['lap_duration'].quantile(0.25)]
    slow_laps = df_analysis[df_analysis['lap_duration'] > df_analysis['lap_duration'].quantile(0.75)]
    
    print(f"\nFast laps (top 25%) - Avg Throttle: {fast_laps['Throttle'].mean():.1f}%, Avg Brake: {fast_laps['brake_total'].mean():.1f}")
    print(f"Slow laps (bottom 25%) - Avg Throttle: {slow_laps['Throttle'].mean():.1f}%, Avg Brake: {slow_laps['brake_total'].mean():.1f}")

# 2. G-force analysis vs fastest laps
print("\n[2] Analyzing cornering G-forces vs lap performance...")
if 'Accel_Lateral' in df_analysis.columns and 'Accel_Longitudinal' in df_analysis.columns:
    df_analysis['total_g_force'] = np.sqrt(
        df_analysis['Accel_Lateral'].fillna(0)**2 + 
        df_analysis['Accel_Longitudinal'].fillna(0)**2
    )
    
    # Find fastest laps per driver
    fastest_laps = df_analysis.loc[df_analysis.groupby('vehicle_number')['lap_duration'].idxmin()]
    
    print(f"\nAverage G-force on fastest laps: {fastest_laps['total_g_force'].mean():.2f}g")
    print(f"Average G-force on all laps: {df_analysis['total_g_force'].mean():.2f}g")
    print(f"Max lateral G-force: {df_analysis['Accel_Lateral'].abs().max():.2f}g")
    print(f"Max longitudinal G-force: {df_analysis['Accel_Longitudinal'].abs().max():.2f}g")

# 3. Weather impact on performance
print("\n[3] Analyzing weather impact on performance...")
if 'AIR_TEMP' in df_analysis.columns:
    # Categorize by temperature
    df_analysis['temp_category'] = pd.cut(df_analysis['AIR_TEMP'], bins=3, labels=['Cool', 'Medium', 'Warm'])
    temp_impact = df_analysis.groupby('temp_category').agg({
        'lap_duration': 'mean',
        'Speed': 'mean',
        'Throttle': 'mean'
    })
    print("\nPerformance by temperature:")
    print(temp_impact)

# ============================================================================
# SECTION 6: AI-Driven Insights & Recommendations
# ============================================================================

print("\n" + "=" * 80)
print("AI-DRIVEN INSIGHTS & RECOMMENDATIONS")
print("=" * 80)

insights = []

# Insight 1: Throttle optimization
if len(df_analysis) > 0 and 'Throttle' in df_analysis.columns:
    optimal_throttle = fast_laps['Throttle'].mean()
    current_avg_throttle = df_analysis['Throttle'].mean()
    
    if current_avg_throttle < optimal_throttle - 5:
        insights.append(f"💡 Drivers could improve lap times by increasing average throttle from {current_avg_throttle:.1f}% to {optimal_throttle:.1f}%")
    elif current_avg_throttle > optimal_throttle + 5:
        insights.append(f"💡 Drivers are over-throttling (avg {current_avg_throttle:.1f}%). Optimal is {optimal_throttle:.1f}% for faster laps")

# Insight 2: Brake pressure optimization
if 'brake_total' in df_analysis.columns:
    optimal_brake = fast_laps['brake_total'].mean()
    current_avg_brake = df_analysis['brake_total'].mean()
    
    if current_avg_brake > optimal_brake * 1.2:
        insights.append(f"💡 Excessive braking detected. Fast laps use {optimal_brake:.1f} bar vs current avg {current_avg_brake:.1f} bar")
        insights.append(f"   → Recommendation: Reduce brake pressure by ~{((current_avg_brake - optimal_brake) / current_avg_brake * 100):.0f}% in technical sections")

# Insight 3: Speed optimization
if 'Speed' in df_analysis.columns:
    optimal_speed = fast_laps['Speed'].mean()
    insights.append(f"💡 Optimal average speed for fast laps: {optimal_speed:.1f} km/h")
    
    # Corner entry speed estimation
    low_speed_sections = df_analysis[df_analysis['Speed'] < df_analysis['Speed'].quantile(0.3)]
    if len(low_speed_sections) > 0:
        optimal_corner_speed = fast_laps[fast_laps['Speed'] < fast_laps['Speed'].quantile(0.3)]['Speed'].mean()
        insights.append(f"💡 Optimal corner entry speed estimated at {optimal_corner_speed:.1f} km/h under current conditions")

# Insight 4: Weather-based recommendations
if 'AIR_TEMP' in df_analysis.columns and 'HUMIDITY' in df_analysis.columns:
    avg_temp = df_analysis['AIR_TEMP'].mean()
    avg_humidity = df_analysis['HUMIDITY'].mean()
    
    if avg_humidity > 70:
        insights.append(f"💡 High humidity ({avg_humidity:.0f}%) detected - reduce throttle aggression in low-grip sections")
    
    if avg_temp < 18:
        insights.append(f"💡 Cool conditions ({avg_temp:.1f}°C) - tires may need extra warm-up laps for optimal grip")

# Insight 5: RPM optimization
if 'Engine_RPM' in df_analysis.columns:
    optimal_rpm = fast_laps['Engine_RPM'].mean()
    insights.append(f"💡 Optimal average RPM for fast laps: {optimal_rpm:.0f} RPM")
    
    max_rpm = df_analysis['Engine_RPM'].max()
    if max_rpm > 7000:
        insights.append(f"   → Some drivers hitting {max_rpm:.0f} RPM - consider earlier upshifts to maintain power band")

# Insight 6: G-force cornering technique
if 'total_g_force' in df_analysis.columns:
    fast_lap_g = fastest_laps['total_g_force'].mean()
    all_lap_g = df_analysis['total_g_force'].mean()
    
    if fast_lap_g > all_lap_g * 1.1:
        insights.append(f"💡 Fastest drivers maintain higher cornering forces ({fast_lap_g:.2f}g vs {all_lap_g:.2f}g avg)")
        insights.append(f"   → Recommendation: Increase corner speed and maintain smoother racing line")

# Print all insights
print("\n")
for i, insight in enumerate(insights, 1):
    print(f"{i}. {insight}")

# ============================================================================
# SECTION 7: Advanced Visualizations
# ============================================================================

print("\n" + "=" * 80)
print("GENERATING VISUALIZATIONS")
print("=" * 80)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)

# ============================================================================
# PLOT 1: Lap Number vs Speed/Acceleration
# ============================================================================

print("\n[1/6] Creating lap number vs speed/acceleration plot...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Telemetry Analysis: Speed & Acceleration Trends', fontsize=16, fontweight='bold')

# Top drivers (by fastest lap)
top_drivers = df_analysis.groupby('vehicle_number')['lap_duration'].min().nsmallest(5).index

# Plot 1a: Speed over laps
ax1 = axes[0, 0]
for driver in top_drivers:
    driver_data = df_analysis[df_analysis['vehicle_number'] == driver]
    ax1.plot(driver_data['lap'], driver_data['Speed'], marker='o', label=f'Car #{driver}', alpha=0.7)
ax1.set_xlabel('Lap Number')
ax1.set_ylabel('Average Speed (km/h)')
ax1.set_title('Speed Progression by Lap (Top 5 Drivers)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 1b: Longitudinal acceleration
ax2 = axes[0, 1]
if 'Accel_Longitudinal' in df_analysis.columns:
    for driver in top_drivers:
        driver_data = df_analysis[df_analysis['vehicle_number'] == driver]
        ax2.plot(driver_data['lap'], driver_data['Accel_Longitudinal'], marker='s', label=f'Car #{driver}', alpha=0.7)
    ax2.set_xlabel('Lap Number')
    ax2.set_ylabel('Longitudinal Acceleration (g)')
    ax2.set_title('Longitudinal G-Force by Lap')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

# Plot 1c: Lateral acceleration
ax3 = axes[1, 0]
if 'Accel_Lateral' in df_analysis.columns:
    for driver in top_drivers:
        driver_data = df_analysis[df_analysis['vehicle_number'] == driver]
        ax3.plot(driver_data['lap'], driver_data['Accel_Lateral'].abs(), marker='^', label=f'Car #{driver}', alpha=0.7)
    ax3.set_xlabel('Lap Number')
    ax3.set_ylabel('Lateral Acceleration (|g|)')
    ax3.set_title('Lateral G-Force by Lap')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

# Plot 1d: Lap time progression
ax4 = axes[1, 1]
for driver in top_drivers:
    driver_data = df_analysis[df_analysis['vehicle_number'] == driver]
    ax4.plot(driver_data['lap'], driver_data['lap_duration'], marker='D', label=f'Car #{driver}', alpha=0.7)
ax4.set_xlabel('Lap Number')
ax4.set_ylabel('Lap Time (seconds)')
ax4.set_title('Lap Time Progression')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('telemetry_analysis/1_speed_acceleration_trends.png', dpi=300, bbox_inches='tight')
print("✓ Saved: telemetry_analysis/1_speed_acceleration_trends.png")
plt.close()

# ============================================================================
# PLOT 2: Heatmap of Throttle vs Brake Usage per Driver
# ============================================================================

print("[2/6] Creating throttle vs brake heatmap...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Driver Performance: Throttle & Brake Analysis', fontsize=16, fontweight='bold')

# Calculate average throttle and brake per driver
driver_throttle_brake = df_analysis.groupby('vehicle_number').agg({
    'Throttle': 'mean',
    'brake_total': 'mean',
    'lap_duration': 'mean'
}).reset_index()

# Sort by lap time
driver_throttle_brake = driver_throttle_brake.sort_values('lap_duration')

# Heatmap 1: Throttle usage
ax1 = axes[0]
throttle_matrix = driver_throttle_brake.pivot_table(
    index='vehicle_number',
    values='Throttle'
).head(20)
sns.heatmap(throttle_matrix, annot=True, fmt='.1f', cmap='RdYlGn', ax=ax1, cbar_kws={'label': 'Throttle %'})
ax1.set_title('Average Throttle Usage by Driver (Top 20)')
ax1.set_xlabel('Throttle %')
ax1.set_ylabel('Car Number')

# Heatmap 2: Brake usage
ax2 = axes[1]
brake_matrix = driver_throttle_brake.pivot_table(
    index='vehicle_number',
    values='brake_total'
).head(20)
sns.heatmap(brake_matrix, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax2, cbar_kws={'label': 'Brake Pressure'})
ax2.set_title('Average Brake Pressure by Driver (Top 20)')
ax2.set_xlabel('Brake Pressure (bar)')
ax2.set_ylabel('Car Number')

plt.tight_layout()
plt.savefig('telemetry_analysis/2_throttle_brake_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Saved: telemetry_analysis/2_throttle_brake_heatmap.png")
plt.close()

# ============================================================================
# PLOT 3: G-Force Analysis
# ============================================================================

print("[3/6] Creating G-force analysis plots...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('G-Force Analysis: Cornering Performance', fontsize=16, fontweight='bold')

if 'Accel_Lateral' in df_analysis.columns and 'Accel_Longitudinal' in df_analysis.columns:
    # Plot 3a: G-G Diagram (Lateral vs Longitudinal)
    ax1 = axes[0, 0]
    scatter = ax1.scatter(df_analysis['Accel_Lateral'], df_analysis['Accel_Longitudinal'],
                         c=df_analysis['Speed'], cmap='viridis', alpha=0.5, s=10)
    plt.colorbar(scatter, ax=ax1, label='Speed (km/h)')
    ax1.set_xlabel('Lateral Acceleration (g)')
    ax1.set_ylabel('Longitudinal Acceleration (g)')
    ax1.set_title('G-G Diagram (colored by speed)')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax1.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
    
    # Plot 3b: G-force distribution
    ax2 = axes[0, 1]
    ax2.hist(df_analysis['total_g_force'], bins=50, color='purple', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Total G-Force (g)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('G-Force Distribution')
    ax2.axvline(df_analysis['total_g_force'].mean(), color='red', linestyle='--', 
                label=f'Mean: {df_analysis["total_g_force"].mean():.2f}g')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3c: G-force vs lap time
    ax3 = axes[1, 0]
    ax3.scatter(df_analysis['total_g_force'], df_analysis['lap_duration'], alpha=0.5, s=20)
    ax3.set_xlabel('Average Total G-Force (g)')
    ax3.set_ylabel('Lap Time (seconds)')
    ax3.set_title('G-Force vs Lap Time Correlation')
    ax3.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(df_analysis['total_g_force'].dropna(), 
                   df_analysis.loc[df_analysis['total_g_force'].notna(), 'lap_duration'], 1)
    p = np.poly1d(z)
    ax3.plot(df_analysis['total_g_force'].sort_values(), 
             p(df_analysis['total_g_force'].sort_values()), 
             "r--", alpha=0.8, label='Trend')
    ax3.legend()
    
    # Plot 3d: Top drivers G-force comparison
    ax4 = axes[1, 1]
    top_driver_g = df_analysis[df_analysis['vehicle_number'].isin(top_drivers)].groupby('vehicle_number').agg({
        'Accel_Lateral': lambda x: x.abs().mean(),
        'Accel_Longitudinal': lambda x: x.abs().mean(),
        'total_g_force': 'mean'
    })
    top_driver_g.plot(kind='bar', ax=ax4, color=['blue', 'orange', 'green'])
    ax4.set_xlabel('Car Number')
    ax4.set_ylabel('Average G-Force (g)')
    ax4.set_title('G-Force Comparison (Top 5 Drivers)')
    ax4.legend(['Lateral', 'Longitudinal', 'Total'])
    ax4.grid(True, alpha=0.3, axis='y')
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=0)

plt.tight_layout()
plt.savefig('telemetry_analysis/3_gforce_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: telemetry_analysis/3_gforce_analysis.png")
plt.close()

# ============================================================================
# PLOT 4: Throttle/Brake Correlation with Lap Times
# ============================================================================

print("[4/6] Creating throttle/brake correlation plots...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Throttle & Brake Impact on Lap Performance', fontsize=16, fontweight='bold')

# Plot 4a: Throttle vs Lap Time
ax1 = axes[0, 0]
ax1.scatter(df_analysis['Throttle'], df_analysis['lap_duration'], alpha=0.5, s=20, c='green')
ax1.set_xlabel('Average Throttle (%)')
ax1.set_ylabel('Lap Time (seconds)')
ax1.set_title('Throttle Usage vs Lap Time')
ax1.grid(True, alpha=0.3)

# Plot 4b: Brake vs Lap Time
ax2 = axes[0, 1]
ax2.scatter(df_analysis['brake_total'], df_analysis['lap_duration'], alpha=0.5, s=20, c='red')
ax2.set_xlabel('Average Brake Pressure (bar)')
ax2.set_ylabel('Lap Time (seconds)')
ax2.set_title('Brake Usage vs Lap Time')
ax2.grid(True, alpha=0.3)

# Plot 4c: Throttle/Brake balance
ax3 = axes[1, 0]
df_analysis['throttle_brake_ratio'] = df_analysis['Throttle'] / (df_analysis['brake_total'] + 1)
ax3.scatter(df_analysis['throttle_brake_ratio'], df_analysis['lap_duration'], alpha=0.5, s=20, c='purple')
ax3.set_xlabel('Throttle/Brake Ratio')
ax3.set_ylabel('Lap Time (seconds)')
ax3.set_title('Throttle-Brake Balance vs Lap Time')
ax3.grid(True, alpha=0.3)

# Plot 4d: Speed vs Throttle
ax4 = axes[1, 1]
scatter = ax4.scatter(df_analysis['Throttle'], df_analysis['Speed'], 
                     c=df_analysis['lap_duration'], cmap='RdYlGn_r', alpha=0.6, s=20)
plt.colorbar(scatter, ax=ax4, label='Lap Time (s)')
ax4.set_xlabel('Average Throttle (%)')
ax4.set_ylabel('Average Speed (km/h)')
ax4.set_title('Throttle vs Speed (colored by lap time)')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('telemetry_analysis/4_throttle_brake_correlation.png', dpi=300, bbox_inches='tight')
print("✓ Saved: telemetry_analysis/4_throttle_brake_correlation.png")
plt.close()

# ============================================================================
# PLOT 5: Weather Impact Visualization
# ============================================================================

print("[5/6] Creating weather impact visualization...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Weather Impact on Performance', fontsize=16, fontweight='bold')

if 'AIR_TEMP' in df_analysis.columns:
    # Plot 5a: Temperature vs Lap Time
    ax1 = axes[0, 0]
    ax1.scatter(df_analysis['AIR_TEMP'], df_analysis['lap_duration'], alpha=0.5, s=20, c='orange')
    ax1.set_xlabel('Air Temperature (°C)')
    ax1.set_ylabel('Lap Time (seconds)')
    ax1.set_title('Temperature vs Lap Time')
    ax1.grid(True, alpha=0.3)
    
    # Plot 5b: Temperature vs Speed
    ax2 = axes[0, 1]
    ax2.scatter(df_analysis['AIR_TEMP'], df_analysis['Speed'], alpha=0.5, s=20, c='red')
    ax2.set_xlabel('Air Temperature (°C)')
    ax2.set_ylabel('Average Speed (km/h)')
    ax2.set_title('Temperature vs Speed')
    ax2.grid(True, alpha=0.3)
    
    # Plot 5c: Humidity vs Performance
    ax3 = axes[1, 0]
    if 'HUMIDITY' in df_analysis.columns:
        ax3.scatter(df_analysis['HUMIDITY'], df_analysis['lap_duration'], alpha=0.5, s=20, c='blue')
        ax3.set_xlabel('Humidity (%)')
        ax3.set_ylabel('Lap Time (seconds)')
        ax3.set_title('Humidity vs Lap Time')
        ax3.grid(True, alpha=0.3)
    
    # Plot 5d: Track temp vs lap time
    ax4 = axes[1, 1]
    if 'TRACK_TEMP' in df_analysis.columns:
        ax4.scatter(df_analysis['TRACK_TEMP'], df_analysis['lap_duration'], alpha=0.5, s=20, c='brown')
        ax4.set_xlabel('Track Temperature (°C)')
        ax4.set_ylabel('Lap Time (seconds)')
        ax4.set_title('Track Temperature vs Lap Time')
        ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('telemetry_analysis/5_weather_impact.png', dpi=300, bbox_inches='tight')
print("✓ Saved: telemetry_analysis/5_weather_impact.png")
plt.close()

# ============================================================================
# PLOT 6: RPM and Gear Analysis
# ============================================================================

print("[6/6] Creating RPM and gear analysis...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Engine Performance: RPM & Gear Analysis', fontsize=16, fontweight='bold')

if 'Engine_RPM' in df_analysis.columns:
    # Plot 6a: RPM distribution
    ax1 = axes[0, 0]
    ax1.hist(df_analysis['Engine_RPM'], bins=50, color='darkblue', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Engine RPM')
    ax1.set_ylabel('Frequency')
    ax1.set_title('RPM Distribution')
    ax1.axvline(df_analysis['Engine_RPM'].mean(), color='red', linestyle='--',
                label=f'Mean: {df_analysis["Engine_RPM"].mean():.0f} RPM')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 6b: RPM vs Speed
    ax2 = axes[0, 1]
    scatter = ax2.scatter(df_analysis['Engine_RPM'], df_analysis['Speed'],
                         c=df_analysis['lap_duration'], cmap='RdYlGn_r', alpha=0.6, s=20)
    plt.colorbar(scatter, ax=ax2, label='Lap Time (s)')
    ax2.set_xlabel('Engine RPM')
    ax2.set_ylabel('Speed (km/h)')
    ax2.set_title('RPM vs Speed (colored by lap time)')
    ax2.grid(True, alpha=0.3)
    
    # Plot 6c: RPM vs Lap Time
    ax3 = axes[1, 0]
    ax3.scatter(df_analysis['Engine_RPM'], df_analysis['lap_duration'], alpha=0.5, s=20, c='purple')
    ax3.set_xlabel('Average Engine RPM')
    ax3.set_ylabel('Lap Time (seconds)')
    ax3.set_title('RPM vs Lap Time')
    ax3.grid(True, alpha=0.3)
    
    # Plot 6d: Gear usage
    ax4 = axes[1, 1]
    if 'Gear' in df_analysis.columns:
        gear_counts = df_analysis['Gear'].value_counts().sort_index()
        ax4.bar(gear_counts.index, gear_counts.values, color='teal', alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Gear')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Gear Usage Distribution')
        ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('telemetry_analysis/6_rpm_gear_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: telemetry_analysis/6_rpm_gear_analysis.png")
plt.close()

# ============================================================================
# SECTION 8: Export Enhanced Data
# ============================================================================

print("\n" + "=" * 80)
print("EXPORTING ENHANCED DATASETS")
print("=" * 80)

# Save merged telemetry data
df_merged.to_csv('telemetry_analysis/merged_telemetry_data.csv', index=False)
print("✓ Saved: telemetry_analysis/merged_telemetry_data.csv")

# Save per-lap statistics
telemetry_wide.to_csv('telemetry_analysis/per_lap_telemetry_stats.csv', index=False)
print("✓ Saved: telemetry_analysis/per_lap_telemetry_stats.csv")

# Save insights to text file
with open('telemetry_analysis/ai_insights.txt', 'w', encoding='utf-8') as f:
    f.write("TOYOTA GR CUP INDIANAPOLIS RACE 1 - AI INSIGHTS\n")
    f.write("=" * 80 + "\n\n")
    for i, insight in enumerate(insights, 1):
        f.write(f"{i}. {insight}\n")
    f.write("\n" + "=" * 80 + "\n")
    f.write(f"Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
print("✓ Saved: telemetry_analysis/ai_insights.txt")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("TELEMETRY ANALYSIS COMPLETE!")
print("=" * 80)

print("\n📊 Generated Visualizations:")
print("  1. telemetry_analysis/1_speed_acceleration_trends.png")
print("  2. telemetry_analysis/2_throttle_brake_heatmap.png")
print("  3. telemetry_analysis/3_gforce_analysis.png")
print("  4. telemetry_analysis/4_throttle_brake_correlation.png")
print("  5. telemetry_analysis/5_weather_impact.png")
print("  6. telemetry_analysis/6_rpm_gear_analysis.png")

print("\n📁 Generated Data Files:")
print("  - telemetry_analysis/merged_telemetry_data.csv")
print("  - telemetry_analysis/per_lap_telemetry_stats.csv")
print("  - telemetry_analysis/ai_insights.txt")

print("\n✨ Key Findings:")
print(f"  - Analyzed {len(df_telemetry):,} telemetry records")
print(f"  - Processed {len(df_analysis)} lap-telemetry combinations")
print(f"  - Generated {len(insights)} AI-driven insights")
print(f"  - Created 6 comprehensive visualization sets")

print("\n🏁 Ready for driver coaching and performance optimization!")
