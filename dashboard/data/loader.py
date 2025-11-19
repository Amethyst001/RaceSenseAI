"""
Data Loading Functions
Handles all CSV loading and data preparation
"""

import pandas as pd
import streamlit as st
import os


def load_data():
    """Load all race data - only drivers with complete ML predictions"""
    try:
        # Load prediction data (ML-based for drivers with enough data)
        if os.path.exists('outputs/predictions/per_driver_predictions.csv'):
            predictions = pd.read_csv('outputs/predictions/per_driver_predictions.csv')
            model_selection = pd.read_csv('models/per_driver_model_selection.csv') if os.path.exists('models/per_driver_model_selection.csv') else None
            enhanced_mode = True
            
            # Filter for drivers with ML predictions
            if 'selected_model' in predictions.columns:
                predictions = predictions[predictions['selected_model'] != 'Basic_Actual'].copy()
            
            print(f"✅ Loaded {len(predictions)} drivers with complete ML predictions")
        else:
            # Use per-driver predictions
            predictions = pd.read_csv('outputs/predictions/per_driver_predictions.csv')
            model_selection = None
            enhanced_mode = False
        
        commentary = pd.read_csv('outputs/commentary/ai_commentary.csv')
        
        try:
            clusters = pd.read_csv('outputs/analysis/driver_clusters.csv')
        except:
            clusters = pd.DataFrame()
        
        # Data validation: Basic sanity checks
        # Lap times should be positive and reasonable (30-300 seconds for this track)
        if 'best_lap' in predictions.columns:
            invalid_best = (predictions['best_lap'] <= 0) | (predictions['best_lap'] > 300)
            if invalid_best.any():
                print(f"⚠️ Warning: {invalid_best.sum()} drivers have invalid best_lap values")
        
        if 'avg_lap' in predictions.columns:
            invalid_avg = (predictions['avg_lap'] <= 0) | (predictions['avg_lap'] > 300)
            if invalid_avg.any():
                print(f"⚠️ Warning: {invalid_avg.sum()} drivers have invalid avg_lap values")
        
        return predictions, commentary, clusters, model_selection, enhanced_mode, True
    except Exception as e:
        st.error(f"⚠️ No data found. Please run: py scripts\\analyze_race.py\\nError: {e}")
        return None, None, None, None, False, False


def load_sector_data():
    """Load sector analysis data"""
    try:
        sector_stats = pd.read_csv('outputs/sector_analysis/sector_statistics.csv')
        sector_data = pd.read_csv('outputs/sector_analysis/sector_data_detailed.csv')
        return sector_stats, sector_data, True
    except:
        return None, None, False


def get_driver_data(predictions, driver_name):
    """Get data for a specific driver"""
    driver_data = predictions[predictions['driver_name'] == driver_name]
    if len(driver_data) > 0:
        return driver_data.iloc[0]
    return None


def get_driver_commentary(commentary, driver_name):
    """Get commentary for a specific driver"""
    return commentary[commentary['driver_name'] == driver_name]


def get_driver_cluster(clusters, driver_name):
    """Get cluster information for a driver"""
    if len(clusters) > 0:
        return clusters[clusters['driver_name'] == driver_name]
    return pd.DataFrame()
