"""
Enhanced Data Loader - Loads all available data sources
Includes weather (wind, pressure), intermediate timing, top speed, flags
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from datetime import datetime


class EnhancedDataLoader:
    """Load and merge all available race data sources"""
    
    def __init__(self, race_number: int = 1):
        self.race_number = race_number
        self.data_path = f"indianapolis/indianapolis/"
        
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """Load all available data sources"""
        print(f"\n[Enhanced Data Loader] Loading Race {self.race_number} data...")
        
        data = {}
        
        # 1. Load enhanced weather data
        data['weather'] = self._load_weather_data()
        print(f"  [OK] Weather data: {len(data['weather'])} records (wind, pressure, rain)")
        
        # 2. Load sector analysis with intermediate timing
        data['sectors'] = self._load_sector_data()
        print(f"  [OK] Sector data: {len(data['sectors'])} laps (6 timing points, top speed, flags)")
        
        # 3. Load telemetry data
        data['telemetry'] = self._load_telemetry_data()
        print(f"  [OK] Telemetry data: {len(data['telemetry'])} records")
        
        # 4. Load lap times
        data['lap_times'] = self._load_lap_times()
        print(f"  [OK] Lap times: {len(data['lap_times'])} laps")
        
        # 5. Load official results
        data['results'] = self._load_official_results()
        print(f"  [OK] Official results: {len(data['results'])} drivers")
        
        # 6. Load best laps
        data['best_laps'] = self._load_best_laps()
        print(f"  [OK] Best laps: {len(data['best_laps'])} entries")
        
        return data
    
    def _load_weather_data(self) -> pd.DataFrame:
        """Load enhanced weather data with wind, pressure, rain"""
        try:
            df = pd.read_csv(
                f"{self.data_path}26_Weather_Race {self.race_number}.CSV",
                sep=';'
            )
            
            # Convert timestamp
            df['TIME_UTC_SECONDS'] = pd.to_numeric(df['TIME_UTC_SECONDS'])
            
            # Clean column names
            df.columns = df.columns.str.strip()
            
            return df
        except Exception as e:
            print(f"  ⚠ Warning: Could not load weather data: {e}")
            return pd.DataFrame()
    
    def _load_sector_data(self) -> pd.DataFrame:
        """Load sector analysis with intermediate timing points"""
        try:
            df = pd.read_csv(
                f"{self.data_path}23_AnalysisEnduranceWithSections_Race {self.race_number}.CSV",
                sep=';'
            )
            
            # Clean column names
            df.columns = df.columns.str.strip()
            
            # Convert lap time to seconds
            df['LAP_TIME_SECONDS'] = df['LAP_TIME'].apply(self._convert_lap_time_to_seconds)
            
            # Convert sector times to seconds
            for col in ['S1_SECONDS', 'S2_SECONDS', 'S3_SECONDS']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Convert intermediate times to seconds
            for col in ['IM1a_time', 'IM1_time', 'IM2a_time', 'IM2_time', 'IM3a_time', 'FL_time']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Convert top speed
            if 'TOP_SPEED' in df.columns:
                df['TOP_SPEED'] = pd.to_numeric(df['TOP_SPEED'], errors='coerce')
            
            return df
        except Exception as e:
            print(f"  ⚠ Warning: Could not load sector data: {e}")
            return pd.DataFrame()
    
    def _load_telemetry_data(self) -> pd.DataFrame:
        """Load telemetry data"""
        try:
            df = pd.read_csv(
                f"{self.data_path}R{self.race_number}_indianapolis_motor_speedway_telemetry.csv"
            )
            return df
        except Exception as e:
            print(f"  ⚠ Warning: Could not load telemetry data: {e}")
            return pd.DataFrame()
    
    def _load_lap_times(self) -> pd.DataFrame:
        """Load lap time data"""
        try:
            df = pd.read_csv(
                f"{self.data_path}R{self.race_number}_indianapolis_motor_speedway_lap_time.csv"
            )
            return df
        except Exception as e:
            print(f"  ⚠ Warning: Could not load lap time data: {e}")
            return pd.DataFrame()
    
    def _load_official_results(self) -> pd.DataFrame:
        """Load official results"""
        try:
            df = pd.read_csv(
                f"{self.data_path}03_GR Cup Race {self.race_number} Official Results.CSV",
                sep=';'
            )
            df.columns = df.columns.str.strip()
            return df
        except Exception as e:
            print(f"  ⚠ Warning: Could not load official results: {e}")
            return pd.DataFrame()
    
    def _load_best_laps(self) -> pd.DataFrame:
        """Load best 10 laps per driver"""
        try:
            df = pd.read_csv(
                f"{self.data_path}99_Best 10 Laps By Driver_Race {self.race_number}.CSV",
                sep=';'
            )
            df.columns = df.columns.str.strip()
            return df
        except Exception as e:
            print(f"  ⚠ Warning: Could not load best laps: {e}")
            return pd.DataFrame()
    
    def _convert_lap_time_to_seconds(self, time_str: str) -> float:
        """Convert lap time string (MM:SS.mmm) to seconds"""
        try:
            if pd.isna(time_str) or time_str == '':
                return np.nan
            
            parts = str(time_str).split(':')
            if len(parts) == 2:
                minutes = int(parts[0])
                seconds = float(parts[1])
                return minutes * 60 + seconds
            else:
                return float(time_str)
        except:
            return np.nan
    
    def merge_weather_with_laps(self, sectors_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
        """Merge weather data with lap data based on timestamp"""
        if weather_df.empty or sectors_df.empty:
            return sectors_df
        
        print("\n[Merging] Matching weather data to laps...")
        
        # Convert elapsed time to unix timestamp (approximate)
        # This is a simplified merge - in production, use actual timestamps
        
        merged = sectors_df.copy()
        
        # For each lap, find closest weather reading
        for idx, lap in merged.iterrows():
            # Simple approach: use average weather for the session
            merged.loc[idx, 'WIND_SPEED'] = weather_df['WIND_SPEED'].mean()
            merged.loc[idx, 'WIND_DIRECTION'] = weather_df['WIND_DIRECTION'].mean()
            merged.loc[idx, 'PRESSURE'] = weather_df['PRESSURE'].mean()
            merged.loc[idx, 'RAIN'] = weather_df['RAIN'].max()
        
        print(f"  [OK] Weather data merged to {len(merged)} laps")
        
        return merged


class IntermediateTimingAnalyzer:
    """Analyze intermediate timing points to pinpoint performance issues"""
    
    # Indianapolis Road Course timing point mapping
    TIMING_POINTS = {
        'IM1a': {'name': 'Turn 1 Entry', 'sector': 1, 'type': 'braking'},
        'IM1': {'name': 'Turn 1 Exit', 'sector': 1, 'type': 'acceleration'},
        'IM2a': {'name': 'Turn 7 Complex Entry', 'sector': 2, 'type': 'technical'},
        'IM2': {'name': 'Turn 7 Complex Exit', 'sector': 2, 'type': 'traction'},
        'IM3a': {'name': 'Oval Entry', 'sector': 3, 'type': 'speed'},
        'FL': {'name': 'Finish Line', 'sector': 3, 'type': 'straight'}
    }
    
    def __init__(self):
        pass
    
    def analyze_driver_timing_points(self, driver_laps: pd.DataFrame) -> Dict:
        """Analyze driver's performance at each timing point"""
        
        if driver_laps.empty:
            return {}
        
        analysis = {
            'timing_points': {},
            'weakest_segments': [],
            'strongest_segments': [],
            'improvement_opportunities': []
        }
        
        # Calculate segments between timing points
        # NOTE: Intermediate times are CUMULATIVE from lap start
        # So segment time = end_time - start_time (NOT start - end)
        segments = [
            ('Start', 'IM1a', 'IM1a_time'),
            ('IM1a', 'IM1', 'IM1_time', 'IM1a_time'),
            ('IM1', 'IM2a', 'IM2a_time', 'IM1_time'),
            ('IM2a', 'IM2', 'IM2_time', 'IM2a_time'),
            ('IM2', 'IM3a', 'IM3a_time', 'IM2_time'),
            ('IM3a', 'FL', 'FL_time', 'IM3a_time')
        ]
        
        for segment in segments:
            if len(segment) == 3:
                # First segment (Start to IM1a) - absolute time
                start, end, time_col = segment
                if time_col in driver_laps.columns:
                    segment_times = driver_laps[time_col].dropna()
                    segment_key = f"{start}_to_{end}"
            else:
                # Other segments (difference between timing points)
                # FIXED: end_time - start_time (was backwards before)
                start, end, end_col, start_col = segment
                if end_col in driver_laps.columns and start_col in driver_laps.columns:
                    segment_times = driver_laps[end_col] - driver_laps[start_col]
                    segment_times = segment_times.dropna()
                    # Filter out negative values (data errors)
                    segment_times = segment_times[segment_times > 0]
                    segment_key = f"{start}_to_{end}"
            
            if len(segment_times) > 0:
                analysis['timing_points'][segment_key] = {
                    'start': start,
                    'end': end,
                    'avg_time': float(segment_times.mean()),
                    'best_time': float(segment_times.min()),
                    'worst_time': float(segment_times.max()),
                    'std_dev': float(segment_times.std()),
                    'potential_gain': float(segment_times.mean() - segment_times.min()),
                    'location': self.TIMING_POINTS.get(end, {}).get('name', end),
                    'type': self.TIMING_POINTS.get(end, {}).get('type', 'unknown')
                }
        
        # Identify weakest segments (highest potential gain)
        sorted_segments = sorted(
            analysis['timing_points'].items(),
            key=lambda x: x[1]['potential_gain'],
            reverse=True
        )
        
        analysis['weakest_segments'] = [
            {
                'segment': seg[0],
                'location': seg[1]['location'],
                'type': seg[1]['type'],
                'potential_gain': seg[1]['potential_gain'],
                'avg_time': seg[1]['avg_time'],
                'best_time': seg[1]['best_time']
            }
            for seg in sorted_segments[:3]
        ]
        
        # Identify strongest segments (most consistent)
        sorted_by_consistency = sorted(
            analysis['timing_points'].items(),
            key=lambda x: x[1]['std_dev']
        )
        
        analysis['strongest_segments'] = [
            {
                'segment': seg[0],
                'location': seg[1]['location'],
                'consistency': seg[1]['std_dev'],
                'avg_time': seg[1]['avg_time']
            }
            for seg in sorted_by_consistency[:3]
        ]
        
        return analysis
    
    def generate_timing_point_recommendations(self, timing_analysis: Dict) -> list:
        """Generate specific recommendations based on timing point analysis"""
        
        recommendations = []
        
        for weakness in timing_analysis.get('weakest_segments', [])[:3]:
            segment_type = weakness['type']
            location = weakness['location']
            gain = weakness['potential_gain']
            
            if gain < 0.05:
                continue
            
            # Type-specific recommendations
            if segment_type == 'braking':
                rec = {
                    'category': 'Braking Zone',
                    'issue': f"Losing {gain:.3f}s at {location}",
                    'action': f"Focus on later, harder braking at {location}. Current average is {weakness['avg_time']:.3f}s vs best of {weakness['best_time']:.3f}s.",
                    'priority': 'HIGH' if gain > 0.15 else 'MEDIUM',
                    'potential_gain': f"{gain:.3f}s per lap",
                    'location': location
                }
            elif segment_type == 'acceleration' or segment_type == 'traction':
                rec = {
                    'category': 'Traction Zone',
                    'issue': f"Losing {gain:.3f}s at {location}",
                    'action': f"Improve throttle application at {location}. Focus on earlier, smoother power delivery.",
                    'priority': 'HIGH' if gain > 0.15 else 'MEDIUM',
                    'potential_gain': f"{gain:.3f}s per lap",
                    'location': location
                }
            elif segment_type == 'technical':
                rec = {
                    'category': 'Technical Section',
                    'issue': f"Losing {gain:.3f}s through {location}",
                    'action': f"Optimize racing line through {location}. Current average is {weakness['avg_time']:.3f}s vs best of {weakness['best_time']:.3f}s.",
                    'priority': 'HIGH' if gain > 0.15 else 'MEDIUM',
                    'potential_gain': f"{gain:.3f}s per lap",
                    'location': location
                }
            else:
                rec = {
                    'category': 'Speed Section',
                    'issue': f"Losing {gain:.3f}s at {location}",
                    'action': f"Carry more speed through {location}. Check for setup or technique issues.",
                    'priority': 'MEDIUM',
                    'potential_gain': f"{gain:.3f}s per lap",
                    'location': location
                }
            
            recommendations.append(rec)
        
        return recommendations


class WindAnalyzer:
    """Analyze wind impact on lap times and sectors"""
    
    def __init__(self):
        # Indianapolis track orientation (approximate)
        self.track_sections = {
            'S1': {'direction': 90, 'length': 'medium', 'exposure': 'high'},  # East
            'S2': {'direction': 180, 'length': 'short', 'exposure': 'medium'},  # South
            'S3': {'direction': 270, 'length': 'long', 'exposure': 'very_high'}  # West (oval)
        }
    
    def analyze_wind_impact(self, laps_df: pd.DataFrame) -> Dict:
        """Analyze how wind affects lap times"""
        
        if 'WIND_SPEED' not in laps_df.columns or 'WIND_DIRECTION' not in laps_df.columns:
            return {}
        
        analysis = {
            'avg_wind_speed': float(laps_df['WIND_SPEED'].mean()),
            'max_wind_speed': float(laps_df['WIND_SPEED'].max()),
            'wind_direction': float(laps_df['WIND_DIRECTION'].mean()),
            'wind_impact_estimate': 0.0,
            'affected_sectors': []
        }
        
        # Estimate wind impact
        avg_wind = analysis['avg_wind_speed']
        
        if avg_wind > 10:
            analysis['wind_impact_estimate'] = 0.15  # High wind
            analysis['severity'] = 'High'
        elif avg_wind > 6:
            analysis['wind_impact_estimate'] = 0.08  # Moderate wind
            analysis['severity'] = 'Moderate'
        else:
            analysis['wind_impact_estimate'] = 0.03  # Low wind
            analysis['severity'] = 'Low'
        
        # Determine which sectors are most affected
        wind_dir = analysis['wind_direction']
        
        for sector, info in self.track_sections.items():
            angle_diff = abs(wind_dir - info['direction'])
            if angle_diff > 180:
                angle_diff = 360 - angle_diff
            
            # Headwind/tailwind effect
            if angle_diff < 45:  # Headwind
                effect = 'Headwind'
                impact = 'negative'
            elif angle_diff > 135:  # Tailwind
                effect = 'Tailwind'
                impact = 'positive'
            else:  # Crosswind
                effect = 'Crosswind'
                impact = 'neutral'
            
            analysis['affected_sectors'].append({
                'sector': sector,
                'effect': effect,
                'impact': impact,
                'exposure': info['exposure']
            })
        
        return analysis
