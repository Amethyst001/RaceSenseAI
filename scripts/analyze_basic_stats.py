"""
Toyota GR Cup Indianapolis Race 1 Data Analysis
Load, merge, analyze, and visualize race data
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# SECTION 1: Load CSV Files Safely
# ============================================================================

print("=" * 80)
print("TOYOTA GR CUP INDIANAPOLIS RACE 1 - DATA ANALYSIS")
print("=" * 80)

# Define file paths
LAP_TIME_FILE = "indianapolis/indianapolis/R1_indianapolis_motor_speedway_lap_time.csv"
LAP_START_FILE = "indianapolis/indianapolis/R1_indianapolis_motor_speedway_lap_start.csv"
LAP_END_FILE = "indianapolis/indianapolis/R1_indianapolis_motor_speedway_lap_end.csv"
WEATHER_FILE = "indianapolis/indianapolis/26_Weather_Race 1.CSV"
RESULTS_FILE = "indianapolis/indianapolis/03_GR Cup Race 1 Official Results.CSV"

# Load lap time data
print("\n[1/5] Loading lap time data...")
try:
    df_laps = pd.read_csv(LAP_TIME_FILE, encoding='utf-8')
    print(f"[OK] Loaded {len(df_laps)} lap records")
except UnicodeDecodeError:
    df_laps = pd.read_csv(LAP_TIME_FILE, encoding='latin-1')
    print(f"[OK] Loaded {len(df_laps)} lap records (latin-1 encoding)")

# Load lap start times
print("\n[2/5] Loading lap start times...")
try:
    df_lap_start = pd.read_csv(LAP_START_FILE, encoding='utf-8')
    print(f"[OK] Loaded {len(df_lap_start)} lap start records")
except UnicodeDecodeError:
    df_lap_start = pd.read_csv(LAP_START_FILE, encoding='latin-1')
    print(f"[OK] Loaded {len(df_lap_start)} lap start records (latin-1 encoding)")

# Load lap end times
print("\n[3/5] Loading lap end times...")
try:
    df_lap_end = pd.read_csv(LAP_END_FILE, encoding='utf-8')
    print(f"[OK] Loaded {len(df_lap_end)} lap end records")
except UnicodeDecodeError:
    df_lap_end = pd.read_csv(LAP_END_FILE, encoding='latin-1')
    print(f"[OK] Loaded {len(df_lap_end)} lap end records (latin-1 encoding)")

# Load weather data (semicolon-separated)
print("\n[4/5] Loading weather data...")
try:
    df_weather = pd.read_csv(WEATHER_FILE, sep=';', encoding='utf-8')
    print(f"? Loaded {len(df_weather)} weather records")
except Exception as e:
    print(f"? Weather file error: {e}")
    df_weather = pd.DataFrame()

# Load official results (semicolon-separated)
print("\n[5/5] Loading official results...")
try:
    df_results = pd.read_csv(RESULTS_FILE, sep=';', encoding='utf-8')
    print(f"? Loaded {len(df_results)} driver results")
except Exception as e:
    print(f"? Results file error: {e}")
    df_results = pd.DataFrame()

# ============================================================================
# SECTION 2: Basic Data Info
# ============================================================================

print("\n" + "=" * 80)
print("DATA OVERVIEW")
print("=" * 80)

# Lap time data info
print(f"\n?? Total laps recorded: {len(df_laps)}")
print(f"?? Unique drivers: {df_laps['vehicle_number'].nunique()}")
print(f"?? Missing values in lap data:")
print(df_laps.isnull().sum())

# Weather data info
if not df_weather.empty:
    print(f"\n???  Weather records: {len(df_weather)}")
    print(f"???  Temperature range: {df_weather['AIR_TEMP'].min():.1f}�C - {df_weather['AIR_TEMP'].max():.1f}�C")

# Results data info
if not df_results.empty:
    print(f"\n?? Drivers in official results: {len(df_results)}")

# ============================================================================
# SECTION 3: Calculate Lap Time Statistics
# ============================================================================

print("\n" + "=" * 80)
print("LAP TIME ANALYSIS")
print("=" * 80)

# Convert timestamps to datetime
df_laps['timestamp'] = pd.to_datetime(df_laps['timestamp'])
df_lap_start['timestamp'] = pd.to_datetime(df_lap_start['timestamp'])
df_lap_end['timestamp'] = pd.to_datetime(df_lap_end['timestamp'])

# Sort by vehicle and lap number
df_laps = df_laps.sort_values(['vehicle_number', 'lap'])

# Remove duplicates BEFORE merging to prevent cartesian product
print("\n?? Removing duplicate lap records...")
initial_start = len(df_lap_start)
initial_end = len(df_lap_end)
initial_laps = len(df_laps)

# Keep first occurrence of each vehicle_number + lap combination
df_lap_start_clean = df_lap_start.drop_duplicates(subset=['vehicle_number', 'lap'], keep='first')
df_lap_end_clean = df_lap_end.drop_duplicates(subset=['vehicle_number', 'lap'], keep='first')
df_laps = df_laps.drop_duplicates(subset=['vehicle_number', 'lap'], keep='first')

print(f"   Lap start: {initial_start} ? {len(df_lap_start_clean)} ({initial_start - len(df_lap_start_clean)} duplicates removed)")
print(f"   Lap end: {initial_end} ? {len(df_lap_end_clean)} ({initial_end - len(df_lap_end_clean)} duplicates removed)")
print(f"   Lap times: {initial_laps} ? {len(df_laps)} ({initial_laps - len(df_laps)} duplicates removed)")

# Merge lap start times
print("\n?? Merging lap start times...")
df_laps = df_laps.merge(
    df_lap_start_clean[['vehicle_number', 'lap', 'timestamp']],
    on=['vehicle_number', 'lap'],
    how='left',
    suffixes=('', '_start')
)

# Merge lap end times
print("?? Merging lap end times...")
df_laps = df_laps.merge(
    df_lap_end_clean[['vehicle_number', 'lap', 'timestamp']],
    on=['vehicle_number', 'lap'],
    how='left',
    suffixes=('', '_end')
)

# Calculate accurate lap duration (end - start)
print("??  Calculating lap durations...")
df_laps['lap_duration'] = (
    df_laps['timestamp_end'] - df_laps['timestamp_start']
).dt.total_seconds()

# Data validation
print("\n? Data Validation:")
initial_count = len(df_laps)
print(f"   Initial laps: {initial_count}")

# Remove laps with missing start/end times
df_laps = df_laps.dropna(subset=['timestamp_start', 'timestamp_end'])
print(f"   After removing missing times: {len(df_laps)} ({initial_count - len(df_laps)} removed)")

# Remove negative lap durations (data errors)
negative_count = (df_laps['lap_duration'] <= 0).sum()
df_laps = df_laps[df_laps['lap_duration'] > 0]
print(f"   After removing negative durations: {len(df_laps)} ({negative_count} removed)")

# Remove unrealistic lap times (< 60s for this track is impossible)
too_fast_count = (df_laps['lap_duration'] < 60).sum()
df_laps = df_laps[df_laps['lap_duration'] >= 60]
print(f"   After removing unrealistic fast laps (<60s): {len(df_laps)} ({too_fast_count} removed)")

# Remove outliers (laps > 200 seconds are likely pit stops or errors)
outlier_count = (df_laps['lap_duration'] >= 200).sum()
df_laps_clean = df_laps[df_laps['lap_duration'] < 200].copy()
print(f"   After removing outliers (>200s): {len(df_laps_clean)} ({outlier_count} removed)")

# Summary statistics
print(f"\n?? Lap Duration Statistics:")
print(f"   Min: {df_laps_clean['lap_duration'].min():.3f}s")
print(f"   Max: {df_laps_clean['lap_duration'].max():.3f}s")
print(f"   Mean: {df_laps_clean['lap_duration'].mean():.3f}s")
print(f"   Median: {df_laps_clean['lap_duration'].median():.3f}s")

# Calculate statistics for each driver
driver_stats = df_laps_clean.groupby('vehicle_number').agg({
    'lap_duration': ['min', 'mean', 'sum', 'count']
}).round(3)

driver_stats.columns = ['fastest_lap', 'avg_lap_time', 'total_race_time', 'laps_completed']
driver_stats = driver_stats.sort_values('fastest_lap')

print("\n???  DRIVER STATISTICS (Top 10 by fastest lap):")
print(driver_stats.head(10).to_string())

# ============================================================================
# SECTION 4: Merge with Official Results
# ============================================================================

print("\n" + "=" * 80)
print("MERGING WITH OFFICIAL RESULTS")
print("=" * 80)

if not df_results.empty:
    # Clean column names
    df_results.columns = df_results.columns.str.strip()
    
    # Merge driver stats with official results by vehicle number
    df_results['NUMBER'] = df_results['NUMBER'].astype(str)
    driver_stats_reset = driver_stats.reset_index()
    driver_stats_reset['vehicle_number'] = driver_stats_reset['vehicle_number'].astype(str)
    
    df_merged = df_results.merge(
        driver_stats_reset,
        left_on='NUMBER',
        right_on='vehicle_number',
        how='left'
    )
    
    # Display merged data
    print("\n?? Merged Results (Position, Driver, Official vs Calculated Times):")
    display_cols = ['POSITION', 'NUMBER', 'DRIVER_FIRSTNAME', 'DRIVER_SECONDNAME', 
                    'LAPS', 'fastest_lap', 'avg_lap_time', 'laps_completed']
    available_cols = [col for col in display_cols if col in df_merged.columns]
    print(df_merged[available_cols].head(15).to_string(index=False))
else:
    df_merged = driver_stats.reset_index()

# ============================================================================
# SECTION 5: Merge with Weather Data
# ============================================================================

print("\n" + "=" * 80)
print("WEATHER CORRELATION")
print("=" * 80)

if not df_weather.empty:
    # Convert weather timestamp
    df_weather['TIME_UTC_SECONDS'] = pd.to_numeric(df_weather['TIME_UTC_SECONDS'], errors='coerce')
    df_weather['weather_time'] = pd.to_datetime(df_weather['TIME_UTC_SECONDS'], unit='s')
    
    # Convert lap timestamp to Unix time for merging
    df_laps_clean['lap_unix_time'] = df_laps_clean['timestamp'].astype(np.int64) // 10**9
    
    # Merge using nearest timestamp (within 60 seconds)
    df_laps_clean = pd.merge_asof(
        df_laps_clean.sort_values('lap_unix_time'),
        df_weather[['TIME_UTC_SECONDS', 'AIR_TEMP', 'TRACK_TEMP', 'HUMIDITY', 'WIND_SPEED']].sort_values('TIME_UTC_SECONDS'),
        left_on='lap_unix_time',
        right_on='TIME_UTC_SECONDS',
        direction='nearest',
        tolerance=60
    )
    
    print("\n???  Weather conditions during race:")
    print(f"   Air Temperature: {df_weather['AIR_TEMP'].mean():.1f}�C (�{df_weather['AIR_TEMP'].std():.1f}�C)")
    print(f"   Track Temperature: {df_weather['TRACK_TEMP'].mean():.1f}�C")
    print(f"   Humidity: {df_weather['HUMIDITY'].mean():.1f}%")
    print(f"   Wind Speed: {df_weather['WIND_SPEED'].mean():.1f} km/h")
    
    # Check correlation between weather and lap times
    if 'AIR_TEMP' in df_laps_clean.columns:
        correlation = df_laps_clean[['lap_duration', 'AIR_TEMP', 'TRACK_TEMP', 'HUMIDITY', 'WIND_SPEED']].corr()['lap_duration']
        print("\n?? Correlation with lap times:")
        print(correlation.to_string())
else:
    print("\n? Weather data not available for correlation analysis")

# ============================================================================
# SECTION 6: Key Insights
# ============================================================================

print("\n" + "=" * 80)
print("KEY INSIGHTS")
print("=" * 80)

# Fastest driver overall
fastest_driver = driver_stats.iloc[0]
print(f"\n?? Fastest lap overall: Car #{fastest_driver.name} - {fastest_driver['fastest_lap']:.3f} seconds")

# Average speed trend (calculate from lap times)
avg_lap_by_lap_num = df_laps_clean.groupby('lap')['lap_duration'].mean()
print(f"\n?? Average lap time trend:")
print(f"   First 5 laps avg: {avg_lap_by_lap_num.head(5).mean():.3f}s")
print(f"   Last 5 laps avg: {avg_lap_by_lap_num.tail(5).mean():.3f}s")
print(f"   Overall average: {avg_lap_by_lap_num.mean():.3f}s")

# Weather impact
if not df_weather.empty and 'AIR_TEMP' in df_laps_clean.columns:
    temp_bins = pd.cut(df_laps_clean['AIR_TEMP'], bins=3, labels=['Cool', 'Medium', 'Warm'])
    lap_time_by_temp = df_laps_clean.groupby(temp_bins)['lap_duration'].mean()
    print(f"\n???  Weather impact on lap times:")
    print(lap_time_by_temp.to_string())

# ============================================================================
# SECTION 7: Visualization
# ============================================================================

print("\n" + "=" * 80)
print("GENERATING VISUALIZATION")
print("=" * 80)

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Toyota GR Cup Indianapolis Race 1 - Analysis', fontsize=16, fontweight='bold')

# Plot 1: Lap times by lap number, colored by weather
ax1 = axes[0, 0]
if not df_weather.empty and 'AIR_TEMP' in df_laps_clean.columns:
    scatter = ax1.scatter(df_laps_clean['lap'], df_laps_clean['lap_duration'], 
                         c=df_laps_clean['AIR_TEMP'], cmap='coolwarm', alpha=0.6, s=20)
    plt.colorbar(scatter, ax=ax1, label='Air Temperature (�C)')
    ax1.set_title('Lap Times vs Lap Number (colored by temperature)')
else:
    ax1.scatter(df_laps_clean['lap'], df_laps_clean['lap_duration'], alpha=0.6, s=20)
    ax1.set_title('Lap Times vs Lap Number')
ax1.set_xlabel('Lap Number')
ax1.set_ylabel('Lap Time (seconds)')
ax1.grid(True, alpha=0.3)

# Plot 2: Average lap time by driver (top 15)
ax2 = axes[0, 1]
top_drivers = driver_stats.head(15)
ax2.barh(range(len(top_drivers)), top_drivers['avg_lap_time'], color='steelblue')
ax2.set_yticks(range(len(top_drivers)))
ax2.set_yticklabels([f"GR86 #{num}" for num in top_drivers.index])
ax2.set_xlabel('Average Lap Time (seconds)')
ax2.set_title('Top 15 Drivers - Average Lap Time')
ax2.invert_yaxis()
ax2.grid(True, alpha=0.3, axis='x')

# Plot 3: Lap time distribution
ax3 = axes[1, 0]
ax3.hist(df_laps_clean['lap_duration'], bins=50, color='green', alpha=0.7, edgecolor='black')
ax3.set_xlabel('Lap Time (seconds)')
ax3.set_ylabel('Frequency')
ax3.set_title('Lap Time Distribution')
ax3.axvline(df_laps_clean['lap_duration'].mean(), color='red', linestyle='--', 
            label=f'Mean: {df_laps_clean["lap_duration"].mean():.2f}s')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Weather conditions over time
ax4 = axes[1, 1]
if not df_weather.empty:
    ax4_twin = ax4.twinx()
    ax4.plot(df_weather.index, df_weather['AIR_TEMP'], 'r-', label='Air Temp', linewidth=2)
    ax4.plot(df_weather.index, df_weather['TRACK_TEMP'], 'orange', label='Track Temp', linewidth=2)
    ax4_twin.plot(df_weather.index, df_weather['HUMIDITY'], 'b--', label='Humidity', linewidth=2)
    ax4.set_xlabel('Time Index')
    ax4.set_ylabel('Temperature (�C)', color='r')
    ax4_twin.set_ylabel('Humidity (%)', color='b')
    ax4.set_title('Weather Conditions During Race')
    ax4.legend(loc='upper left')
    ax4_twin.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
else:
    ax4.text(0.5, 0.5, 'Weather data not available', 
             ha='center', va='center', transform=ax4.transAxes, fontsize=12)
    ax4.set_title('Weather Conditions')

plt.tight_layout()
plt.savefig('outputs/analysis/race1_analysis.png', dpi=300, bbox_inches='tight')
print("\n? Visualization saved as 'outputs/analysis/race1_analysis.png'")

# ============================================================================
# SECTION 8: Export Summary
# ============================================================================

print("\n" + "=" * 80)
print("EXPORTING SUMMARY")
print("=" * 80)

# Create output directory
import os
os.makedirs('outputs/analysis', exist_ok=True)

# Save driver statistics to CSV
driver_stats.to_csv('outputs/analysis/driver_statistics.csv')
print("? Driver statistics saved to 'outputs/analysis/driver_statistics.csv'")

# Save merged results if available
if not df_merged.empty:
    df_merged.to_csv('outputs/analysis/merged_results.csv', index=False)
    print("? Merged results saved to 'outputs/analysis/merged_results.csv'")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)
print("\nGenerated files in outputs/analysis/:")
print("  - race1_analysis.png (visualization)")
print("  - driver_statistics.csv (lap time stats)")
print("  - merged_results.csv (combined data)")

