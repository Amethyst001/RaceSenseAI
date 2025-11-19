"""
Enhanced Data Processing for AI Analysis
Prepares and enriches data for intelligent analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import json
import os

class EnhancedDataProcessor:
    """
    Processes raw racing data into enriched format for AI analysis
    """
    
    def __init__(self):
        self.cache = {}
        self.benchmark_data = {}
        
    def load_and_process_driver_data(self, driver_name: str) -> Tuple[pd.DataFrame, pd.DataFrame, Dict, Dict]:
        """Load and process all data for a specific driver"""
        
        # Load prediction data
        predictions = self._load_predictions_data(driver_name)
        
        # Load telemetry
        telemetry_data = self._load_telemetry_data(driver_name)
        
        # Load driver performance data
        driver_data = self._load_driver_performance_data(driver_name)
        
        # Load benchmark data
        benchmark_data = self._load_benchmark_data()
        
        return driver_data, telemetry_data, predictions, benchmark_data
    
    def _load_predictions_data(self, driver_name: str) -> Dict:
        """Load prediction data for driver"""
        try:
            # Try enhanced predictions first
            predictions_df = pd.read_csv('outputs/predictions/per_driver_predictions.csv')
        except:
            try:
                predictions_df = pd.read_csv('outputs/predictions/predicted_lap_times.csv')
            except:
                return self._create_default_predictions()
        
        # Filter for specific driver
        driver_predictions = predictions_df[predictions_df['driver_name'] == driver_name]
        
        if len(driver_predictions) == 0:
            return self._create_default_predictions()
        
        row = driver_predictions.iloc[0]
        
        return {
            'predicted_lap': row.get('predicted_next_lap', row.get('predicted_lap', 0)),
            'confidence_score': self._normalize_confidence(row.get('confidence_score', row.get('confidence', 50))),
            'models_in_agreement': row.get('models_in_agreement', 3),
            'mae': row.get('mae', 1.0),
            'coverage': row.get('coverage', 0.85),
            'selected_model': row.get('selected_model', 'Unknown'),
            'prediction_lower': row.get('prediction_lower', 0),
            'prediction_upper': row.get('prediction_upper', 0)
        }
    
    def _normalize_confidence(self, confidence) -> float:
        """Normalize confidence score to 0-100 range"""
        if isinstance(confidence, str):
            if 'Low' in confidence:
                return 30
            elif 'Medium' in confidence:
                return 60
            elif 'High' in confidence:
                return 85
            else:
                return 50
        return float(confidence) if confidence else 50
    
    def _load_telemetry_data(self, driver_name: str) -> pd.DataFrame:
        """Load telemetry data for driver"""
        try:
            telemetry_df = pd.read_csv('telemetry_analysis/merged_telemetry_data.csv')
        except:
            return pd.DataFrame()
        
        # Extract driver number from driver name
        try:
            if '#' in driver_name:
                driver_num = int(driver_name.split('#')[1])
            else:
                # Extract number from string
                import re
                numbers = re.findall(r'\d+', driver_name)
                driver_num = int(numbers[0]) if numbers else 0
        except:
            return pd.DataFrame()
        
        # Filter telemetry for this driver
        driver_telemetry = telemetry_df[telemetry_df['vehicle_number'] == driver_num]
        
        return driver_telemetry
    
    def _load_driver_performance_data(self, driver_name: str) -> pd.DataFrame:
        """Load driver performance data"""
        try:
            # Load from predictions file
            predictions_df = pd.read_csv('outputs/predictions/per_driver_predictions.csv')
            driver_data = predictions_df[predictions_df['driver_name'] == driver_name]
            
            if len(driver_data) > 0:
                return driver_data
        except:
            pass
        
        # Create minimal driver data
        driver_num = 0
        if '#' in driver_name:
            try:
                driver_num = int(driver_name.split('#')[1])
            except:
                pass
        
        return pd.DataFrame([{
            'driver_name': driver_name,
            'driver_id': driver_num,
            'best_lap': 0,
            'avg_lap': 0
        }])
    
    def _load_benchmark_data(self) -> Dict:
        """Load benchmark data for comparative analysis"""
        if 'benchmark' in self.cache:
            return self.cache['benchmark']
        
        try:
            # Load telemetry for fastest lap analysis
            telemetry_df = pd.read_csv('telemetry_analysis/merged_telemetry_data.csv')
            
            if not telemetry_df.empty and 'lap_duration' in telemetry_df.columns:
                fastest_lap = telemetry_df['lap_duration'].min()
                session_average = telemetry_df['lap_duration'].mean()
                
                benchmark = {
                    'fastest_lap': fastest_lap,
                    'session_average': session_average,
                    'fastest_driver': telemetry_df.loc[telemetry_df['lap_duration'].idxmin(), 'vehicle_number']
                }
            else:
                benchmark = self._create_default_benchmark()
        except:
            benchmark = self._create_default_benchmark()
        
        self.cache['benchmark'] = benchmark
        return benchmark
    
    def _create_default_predictions(self) -> Dict:
        """Create default prediction data when none available"""
        return {
            'predicted_lap': 100.0,
            'confidence_score': 50,
            'models_in_agreement': 3,
            'mae': 1.0,
            'coverage': 0.85,
            'selected_model': 'Default',
            'prediction_lower': 99.0,
            'prediction_upper': 101.0
        }
    
    def _create_default_benchmark(self) -> Dict:
        """Create default benchmark data"""
        return {
            'fastest_lap': 98.0,
            'session_average': 102.0,
            'fastest_driver': 0
        }
    
    def enrich_telemetry_data(self, telemetry_df: pd.DataFrame) -> pd.DataFrame:
        """Enrich telemetry data with calculated metrics"""
        if telemetry_df.empty:
            return telemetry_df
        
        enriched_df = telemetry_df.copy()
        
        # Calculate additional metrics
        if 'Speed' in enriched_df.columns:
            enriched_df['speed_consistency'] = enriched_df['Speed'].rolling(window=10).std()
            enriched_df['speed_efficiency'] = enriched_df['Speed'] / enriched_df['Speed'].max()
        
        if 'Throttle' in enriched_df.columns:
            enriched_df['throttle_smoothness'] = 1 - (enriched_df['Throttle'].rolling(window=5).std() / 100)
            enriched_df['throttle_efficiency'] = enriched_df['Throttle'] / 100
        
        if 'Brake_Front' in enriched_df.columns:
            enriched_df['brake_consistency'] = 1 - (enriched_df['Brake_Front'].rolling(window=5).std() / 100)
        
        # Calculate sector performance if lap data available
        if 'lap_duration' in enriched_df.columns:
            enriched_df['lap_rank'] = enriched_df['lap_duration'].rank()
            enriched_df['pace_delta'] = enriched_df['lap_duration'] - enriched_df['lap_duration'].min()
        
        return enriched_df
    
    def calculate_performance_metrics(self, telemetry_df: pd.DataFrame) -> Dict:
        """Calculate comprehensive performance metrics"""
        if telemetry_df.empty:
            return {}
        
        metrics = {}
        
        # Speed metrics
        if 'Speed' in telemetry_df.columns:
            metrics['speed'] = {
                'average': telemetry_df['Speed'].mean(),
                'maximum': telemetry_df['Speed'].max(),
                'consistency': 1 - (telemetry_df['Speed'].std() / telemetry_df['Speed'].mean()),
                'efficiency': telemetry_df['Speed'].mean() / telemetry_df['Speed'].max()
            }
        
        # Throttle metrics
        if 'Throttle' in telemetry_df.columns:
            metrics['throttle'] = {
                'average': telemetry_df['Throttle'].mean(),
                'maximum': telemetry_df['Throttle'].max(),
                'smoothness': 1 - (telemetry_df['Throttle'].std() / 100),
                'full_throttle_pct': (telemetry_df['Throttle'] >= 95).sum() / len(telemetry_df) * 100
            }
        
        # Braking metrics
        if 'Brake_Front' in telemetry_df.columns:
            metrics['braking'] = {
                'average': telemetry_df['Brake_Front'].mean(),
                'maximum': telemetry_df['Brake_Front'].max(),
                'consistency': 1 - (telemetry_df['Brake_Front'].std() / 100),
                'efficiency': self._calculate_braking_efficiency(telemetry_df)
            }
        
        # Lap time metrics
        if 'lap_duration' in telemetry_df.columns:
            lap_times = telemetry_df['lap_duration'].dropna()
            if len(lap_times) > 0:
                metrics['lap_times'] = {
                    'best': lap_times.min(),
                    'average': lap_times.mean(),
                    'consistency': 1 - (lap_times.std() / lap_times.mean()),
                    'improvement_trend': self._calculate_improvement_trend(lap_times)
                }
        
        return metrics
    
    def _calculate_braking_efficiency(self, telemetry_df: pd.DataFrame) -> float:
        """Calculate braking efficiency metric"""
        if 'Brake_Front' in telemetry_df.columns and 'Speed' in telemetry_df.columns:
            # Simple efficiency: less braking for same speed reduction
            brake_events = telemetry_df[telemetry_df['Brake_Front'] > 10]
            if len(brake_events) > 0:
                return 1 - (brake_events['Brake_Front'].mean() / 100)
        return 0.5
    
    def _calculate_improvement_trend(self, lap_times: pd.Series) -> str:
        """Calculate if driver is improving, maintaining, or declining"""
        if len(lap_times) < 3:
            return "insufficient_data"
        
        # Compare first third vs last third
        first_third = lap_times.head(len(lap_times)//3).mean()
        last_third = lap_times.tail(len(lap_times)//3).mean()
        
        improvement = first_third - last_third
        
        if improvement > 0.5:
            return "improving"
        elif improvement < -0.5:
            return "declining"
        else:
            return "stable"
    
    def prepare_session_context(self) -> Dict:
        """Prepare session context information"""
        # This would ideally come from session metadata
        # For now, create reasonable defaults
        
        return {
            'session_type': 'practice',  # practice, qualifying, race
            'session_phase': 'mid',      # early, mid, late
            'weather_conditions': {
                'track_temp': 25.0,
                'air_temp': 22.0,
                'humidity': 65.0,
                'wind_speed': 5.0
            },
            'tire_condition': 'good',    # new, good, worn, critical
            'traffic_density': 0.3,      # 0-1 scale
            'track_evolution': 0.2       # 0-1 scale, how much track improved
        }