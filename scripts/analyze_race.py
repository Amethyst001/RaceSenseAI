"""
Toyota GR Cup Indianapolis Race 1 - Master Analysis Script
Orchestrates all analysis phases in sequence
"""

import subprocess
import sys
import os
from datetime import datetime

print("=" * 80)
print("TOYOTA GR CUP INDIANAPOLIS RACE 1 - MASTER ANALYSIS")
print("=" * 80)
print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\nThis will run all analysis steps:")
print("  [1/4] Load and analyze race data (10 seconds)")
print("  [2/4] Process telemetry (2-3 minutes)")
print("  [3/4] Generate dashboards (1-2 minutes)")
print("  [4/4] ML predictions & AI commentary (1-2 minutes)")
print("\nTotal estimated time: 5-7 minutes")
print("=" * 80)

def run_phase(phase_num, script_name, description):
    """Run a phase script and handle errors"""
    print(f"\n{'=' * 80}")
    print(f"[{phase_num}/4] {description}")
    print(f"Running: {script_name}")
    print('=' * 80)
    
    try:
        result = subprocess.run(
            [sys.executable, f"scripts/{script_name}"],
            check=True,
            capture_output=False,
            text=True
        )
        print(f"✓ Phase {phase_num} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Phase {phase_num} failed with error code {e.returncode}")
        print(f"Error: {e}")
        return False
    except Exception as e:
        print(f"✗ Phase {phase_num} failed: {e}")
        return False

# Create output structure
os.makedirs('outputs/analysis', exist_ok=True)
os.makedirs('outputs/predictions', exist_ok=True)
os.makedirs('outputs/visualizations', exist_ok=True)
os.makedirs('outputs/commentary', exist_ok=True)
os.makedirs('outputs/reports', exist_ok=True)

# ============================================================================
# RUN ALL PHASES
# ============================================================================

# Run basic analysis
success = run_phase(1, "analyze_basic_stats.py", "Load and analyze race data")
if not success:
    print("\n❌ Analysis stopped due to basic analysis failure")
    sys.exit(1)

# Telemetry processing (skipped by default - takes 2-3 minutes)
print("\n" + "=" * 80)
print("Telemetry Processing (SKIPPED - takes 2-3 minutes)")
print("To run telemetry processing, execute: py scripts/process_telemetry.py")
print("=" * 80)

# Dashboard generation (skipped - using interactive dashboard)
print("\n" + "=" * 80)
print("Static Dashboard Generation (SKIPPED - using interactive dashboard)")
print("=" * 80)

# Run predictions & AI commentary (required)
success = run_phase(4, "generate_predictions.py", "ML predictions & AI commentary")
if not success:
    print("\n❌ Analysis stopped due to prediction failure")
    print("Predictions are required for the dashboard to work.")
    sys.exit(1)

# ============================================================================
# GENERATE SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("GENERATING FINAL SUMMARY")
print("=" * 80)


# Verify predictions completed successfully
import pandas as pd

try:
    predictions = pd.read_csv('outputs/predictions/predicted_lap_times.csv')
    commentary = pd.read_csv('outputs/predictions/ai_commentary_by_driver.csv')
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n✅ All phases completed successfully!")
    print(f"\n📊 Key Results:")
    print(f"   - {len(predictions)} drivers analyzed")
    print(f"   - Top prediction: {predictions.iloc[0]['driver_name']} ({predictions.iloc[0]['predicted_next_lap']:.3f}s)")
    print(f"   - {len(commentary)} commentary insights generated")
    print(f"\n📁 Outputs:")
    print(f"   - Analysis: outputs/analysis/")
    print(f"   - Predictions: outputs/predictions/")
    if os.path.exists('telemetry_analysis'):
        print(f"   - Telemetry: telemetry_analysis/")
    print(f"\n🚀 Next Steps:")
    print(f"   1. Launch dashboard: py -m streamlit run dashboard/dashboard.py")
    print(f"   2. Or use: run_full_analysis.bat (includes dashboard launch)")
    print(f"   3. Check predictions: outputs/predictions/predicted_lap_times.csv")
    print(f"   4. Read commentary: outputs/predictions/ai_commentary_summary.txt")
    print("\n" + "=" * 80)
    
except Exception as e:
    print(f"\n⚠️  Could not load final results: {e}")
    print("Check individual phase outputs for details.")
