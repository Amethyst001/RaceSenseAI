"""
Toyota GR Cup Indianapolis Race 1 - Sector Analysis
Analyze sector times (S1, S2, S3) for detailed performance insights
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
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

print("=" * 80)
print("TOYOTA GR CUP INDIANAPOLIS RACE 1 - SECTOR ANALYSIS")
print("=" * 80)

# Create output directory
os.makedirs('outputs/sector_analysis', exist_ok=True)

# Load sector data from endurance analysis file
SECTOR_FILE = "indianapolis/indianapolis/23_AnalysisEnduranceWithSections_Race 1.CSV"

print("\n[1/3] Loading sector data...")
try:
    df_sectors = pd.read_csv(SECTOR_FILE, sep=';', encoding='utf-8')
    print(f"[OK] Loaded {len(df_sectors)} sector records")
except FileNotFoundError:
    print(f"[ERROR] Sector file not found: {SECTOR_FILE}")
    print("[INFO] Attempting to use telemetry data as fallback...")
    try:
        telemetry = pd.read_csv('telemetry_analysis/merged_telemetry_data.csv')
        print(f"[OK] Using telemetry data with {len(telemetry)} records")
        # Create synthetic sector data from telemetry if needed
        df_sectors = None
    except:
        print("[ERROR] No data source available. Exiting.")
        exit(1)

# Clean column names (remove leading/trailing spaces)
df_sectors.columns = df_sectors.columns.str.strip()

print(f"\n[DATA] Columns available: {df_sectors.columns.tolist()[:10]}...")
print(f"[DATA] Unique drivers: {df_sectors['NUMBER'].nunique()}")

# Use NUMBER as the driver identifier (not DRIVER_NUMBER which is always 1)
df_sectors['driver_number'] = df_sectors['NUMBER']

# Load telemetry for additional insights
print("\n[2/3] Loading telemetry data for enhanced analysis...")
try:
    telemetry = pd.read_csv('telemetry_analysis/merged_telemetry_data.csv')
    has_telemetry = True
    print(f"[OK] Loaded {len(telemetry)} telemetry records")
except:
    has_telemetry = False
    print("[WARN] Telemetry data not available")

# Convert sector times to numeric (they're in seconds)
print("\n[3/3] Processing sector times...")

# Check which column format we have
if 'S1_SECONDS' in df_sectors.columns:
    sector_cols = ['S1_SECONDS', 'S2_SECONDS', 'S3_SECONDS']
elif 'S1' in df_sectors.columns:
    sector_cols = ['S1', 'S2', 'S3']
    # Rename for consistency
    df_sectors.rename(columns={'S1': 'S1_SECONDS', 'S2': 'S2_SECONDS', 'S3': 'S3_SECONDS'}, inplace=True)
else:
    print("[ERROR] No sector columns found")
    exit(1)

for col in ['S1_SECONDS', 'S2_SECONDS', 'S3_SECONDS']:
    if col in df_sectors.columns:
        df_sectors[col] = pd.to_numeric(df_sectors[col], errors='coerce')

# Remove rows with missing sector data
initial_count = len(df_sectors)
df_sectors = df_sectors.dropna(subset=['S1_SECONDS', 'S2_SECONDS', 'S3_SECONDS'])
print(f"[OK] Valid sector records: {len(df_sectors)} ({initial_count - len(df_sectors)} removed)")

# Calculate total lap time from sectors
df_sectors['CALCULATED_LAP_TIME'] = (
    df_sectors['S1_SECONDS'] + 
    df_sectors['S2_SECONDS'] + 
    df_sectors['S3_SECONDS']
)

# Data validation: remove unrealistic sector times
print("\n[VALIDATION] Removing unrealistic sector times...")
initial = len(df_sectors)

# S1, S2, S3 should each be 20-80 seconds for this track
df_sectors = df_sectors[
    (df_sectors['S1_SECONDS'] > 20) & (df_sectors['S1_SECONDS'] < 80) &
    (df_sectors['S2_SECONDS'] > 20) & (df_sectors['S2_SECONDS'] < 80) &
    (df_sectors['S3_SECONDS'] > 20) & (df_sectors['S3_SECONDS'] < 80)
]
print(f"[OK] After validation: {len(df_sectors)} ({initial - len(df_sectors)} removed)")

# ============================================================================
# SECTION 1: Per-Driver Sector Analysis
# ============================================================================

print("\n" + "=" * 80)
print("PER-DRIVER SECTOR ANALYSIS")
print("=" * 80)

# Calculate best, average, and consistency for each driver per sector
sector_stats = df_sectors.groupby('driver_number').agg({
    'S1_SECONDS': ['min', 'mean', 'std', 'count'],
    'S2_SECONDS': ['min', 'mean', 'std', 'count'],
    'S3_SECONDS': ['min', 'mean', 'std', 'count'],
    'CALCULATED_LAP_TIME': ['min', 'mean', 'std'],
    'TOP_SPEED': 'max'
}).round(3)

# Flatten column names
sector_stats.columns = ['_'.join(col).strip() for col in sector_stats.columns.values]
sector_stats = sector_stats.reset_index()

# Rename for clarity
sector_stats.columns = [
    'driver_number',
    's1_best', 's1_avg', 's1_std', 's1_laps',
    's2_best', 's2_avg', 's2_std', 's2_laps',
    's3_best', 's3_avg', 's3_std', 's3_laps',
    'lap_best', 'lap_avg', 'lap_std',
    'top_speed'
]

# Calculate sector consistency scores (lower std = more consistent)
sector_stats['s1_consistency'] = (1 - (sector_stats['s1_std'] / sector_stats['s1_avg'])) * 100
sector_stats['s2_consistency'] = (1 - (sector_stats['s2_std'] / sector_stats['s2_avg'])) * 100
sector_stats['s3_consistency'] = (1 - (sector_stats['s3_std'] / sector_stats['s3_avg'])) * 100
sector_stats['overall_consistency'] = (
    sector_stats['s1_consistency'] + 
    sector_stats['s2_consistency'] + 
    sector_stats['s3_consistency']
) / 3

# Identify best sector for each driver (where they lose least time vs overall best)
overall_best_s1 = sector_stats['s1_best'].min()
overall_best_s2 = sector_stats['s2_best'].min()
overall_best_s3 = sector_stats['s3_best'].min()

sector_stats['s1_gap_to_best'] = sector_stats['s1_best'] - overall_best_s1
sector_stats['s2_gap_to_best'] = sector_stats['s2_best'] - overall_best_s2
sector_stats['s3_gap_to_best'] = sector_stats['s3_best'] - overall_best_s3

# Determine strongest and weakest sector for each driver
def get_strongest_sector(row):
    gaps = {
        'S1': row['s1_gap_to_best'],
        'S2': row['s2_gap_to_best'],
        'S3': row['s3_gap_to_best']
    }
    return min(gaps, key=gaps.get)

def get_weakest_sector(row):
    gaps = {
        'S1': row['s1_gap_to_best'],
        'S2': row['s2_gap_to_best'],
        'S3': row['s3_gap_to_best']
    }
    return max(gaps, key=gaps.get)

sector_stats['strongest_sector'] = sector_stats.apply(get_strongest_sector, axis=1)
sector_stats['weakest_sector'] = sector_stats.apply(get_weakest_sector, axis=1)
sector_stats['improvement_potential'] = sector_stats[[
    's1_gap_to_best', 's2_gap_to_best', 's3_gap_to_best'
]].max(axis=1)

# Sort by best lap time
sector_stats = sector_stats.sort_values('lap_best')

print("\n[TOP 10] Drivers by Best Lap Time:")
print(sector_stats[[
    'driver_number', 'lap_best', 's1_best', 's2_best', 's3_best', 
    'strongest_sector', 'weakest_sector', 'improvement_potential'
]].head(10).to_string(index=False))

# ============================================================================
# SECTION 2: Sector-Specific Insights
# ============================================================================

print("\n" + "=" * 80)
print("SECTOR-SPECIFIC INSIGHTS")
print("=" * 80)

print(f"\n[SECTOR 1] Best: {overall_best_s1:.3f}s | Avg: {sector_stats['s1_avg'].mean():.3f}s")
print(f"[SECTOR 2] Best: {overall_best_s2:.3f}s | Avg: {sector_stats['s2_avg'].mean():.3f}s")
print(f"[SECTOR 3] Best: {overall_best_s3:.3f}s | Avg: {sector_stats['s3_avg'].mean():.3f}s")

# Identify which sector has most variation (hardest to master)
sector_variations = {
    'S1': sector_stats['s1_std'].mean(),
    'S2': sector_stats['s2_std'].mean(),
    'S3': sector_stats['s3_std'].mean()
}
hardest_sector = max(sector_variations, key=sector_variations.get)
print(f"\n[INSIGHT] Hardest sector to master: {hardest_sector} (highest variation: {sector_variations[hardest_sector]:.3f}s)")

# Count strongest/weakest sectors
print(f"\n[INSIGHT] Sector strengths across drivers:")
print(f"  S1 strongest for: {(sector_stats['strongest_sector'] == 'S1').sum()} drivers")
print(f"  S2 strongest for: {(sector_stats['strongest_sector'] == 'S2').sum()} drivers")
print(f"  S3 strongest for: {(sector_stats['strongest_sector'] == 'S3').sum()} drivers")

# ============================================================================
# SECTION 3: Save Results
# ============================================================================

print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

# Save sector statistics
sector_stats.to_csv('outputs/sector_analysis/sector_statistics.csv', index=False)
print("[OK] Sector statistics saved to 'outputs/sector_analysis/sector_statistics.csv'")

# Save detailed sector data
df_sectors.to_csv('outputs/sector_analysis/sector_data_detailed.csv', index=False)
print("[OK] Detailed sector data saved to 'outputs/sector_analysis/sector_data_detailed.csv'")

# ============================================================================
# SECTION 4: Visualization
# ============================================================================

print("\n" + "=" * 80)
print("GENERATING VISUALIZATIONS")
print("=" * 80)

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Toyota GR Cup Indianapolis - Sector Analysis', fontsize=16, fontweight='bold')

# Plot 1: Best sector times by driver (top 15)
ax1 = axes[0, 0]
top_15 = sector_stats.head(15)
x = np.arange(len(top_15))
width = 0.25

ax1.bar(x - width, top_15['s1_best'], width, label='Sector 1', color='#EB0A1E', alpha=0.8)
ax1.bar(x, top_15['s2_best'], width, label='Sector 2', color='#FFD700', alpha=0.8)
ax1.bar(x + width, top_15['s3_best'], width, label='Sector 3', color='#00BFFF', alpha=0.8)

ax1.set_xlabel('Driver Number', fontweight='bold')
ax1.set_ylabel('Best Sector Time (seconds)', fontweight='bold')
ax1.set_title('Best Sector Times - Top 15 Drivers')
ax1.set_xticks(x)
ax1.set_xticks(x)
ax1.set_xticklabels(top_15['driver_number'].astype(int), rotation=45)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Plot 2: Sector consistency (top 15)
ax2 = axes[0, 1]
ax2.barh(range(len(top_15)), top_15['overall_consistency'], color='#28A745', alpha=0.7)
ax2.set_yticks(range(len(top_15)))
ax2.set_yticklabels(top_15['driver_number'].astype(int))
ax2.set_xlabel('Consistency Score (%)', fontweight='bold')
ax2.set_ylabel('Driver Number', fontweight='bold')
ax2.set_title('Overall Sector Consistency - Top 15 Drivers')
ax2.grid(axis='x', alpha=0.3)
ax2.invert_yaxis()

# Plot 3: Improvement potential by sector
ax3 = axes[1, 0]
improvement_data = sector_stats.head(15)[['s1_gap_to_best', 's2_gap_to_best', 's3_gap_to_best']]
improvement_data.plot(kind='bar', ax=ax3, color=['#EB0A1E', '#FFD700', '#00BFFF'], alpha=0.7)
ax3.set_xlabel('Driver Number', fontweight='bold')
ax3.set_ylabel('Gap to Best (seconds)', fontweight='bold')
ax3.set_title('Improvement Potential by Sector - Top 15 Drivers')
ax3.set_xticklabels(top_15['driver_number'].astype(int), rotation=45)
ax3.legend(['Sector 1', 'Sector 2', 'Sector 3'])
ax3.grid(axis='y', alpha=0.3)

# Plot 4: Sector distribution (box plot)
ax4 = axes[1, 1]
sector_data = [
    df_sectors['S1_SECONDS'].dropna(),
    df_sectors['S2_SECONDS'].dropna(),
    df_sectors['S3_SECONDS'].dropna()
]
bp = ax4.boxplot(sector_data, labels=['Sector 1', 'Sector 2', 'Sector 3'],
                 patch_artist=True, showmeans=True)
colors = ['#EB0A1E', '#FFD700', '#00BFFF']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax4.set_ylabel('Time (seconds)', fontweight='bold')
ax4.set_title('Sector Time Distribution (All Laps)')
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/sector_analysis/sector_analysis.png', dpi=300, bbox_inches='tight')
print("[OK] Visualization saved to 'outputs/sector_analysis/sector_analysis.png'")

print("\n" + "=" * 80)
print("SECTOR ANALYSIS COMPLETE!")
print("=" * 80)
print("\nGenerated files in outputs/sector_analysis/:")
print("  - sector_statistics.csv (per-driver sector stats)")
print("  - sector_data_detailed.csv (all sector data)")
print("  - sector_analysis.png (visualizations)")
print("\nKey Insights:")
print(f"  - Fastest overall lap: {sector_stats.iloc[0]['lap_best']:.3f}s (Driver #{int(sector_stats.iloc[0]['driver_number'])})")
print(f"  - Hardest sector: {hardest_sector}")
print(f"  - Average improvement potential: {sector_stats['improvement_potential'].mean():.3f}s")
