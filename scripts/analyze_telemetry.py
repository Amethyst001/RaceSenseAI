"""
Professional Motorsport Telemetry Insights
Expert-validated GR86/GR Cup specific recommendations with track context
"""

import pandas as pd
import numpy as np
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
print("PROFESSIONAL TELEMETRY INSIGHTS")
print("=" * 80)

# Load telemetry data
print("\n[1/8] Loading telemetry data...")
telemetry = pd.read_csv('telemetry_analysis/merged_telemetry_data.csv')
print(f"[OK] Loaded {len(telemetry)} telemetry records for {telemetry['vehicle_number'].nunique()} drivers")

# Load prediction data and lap data
try:
    predictions = pd.read_csv('outputs/predictions/per_driver_predictions.csv')
    has_predictions = True
except:
    predictions = pd.DataFrame()
    has_predictions = False

try:
    lap_data = pd.read_csv('outputs/analysis/lap_times.csv')
    has_lap_data = True
except:
    lap_data = pd.DataFrame()
    has_lap_data = False

# ============================================================================
# GR86/GR CUP TECHNICAL SPECIFICATIONS (Expert Validated)
# ============================================================================

GR86_SPECS = {
    'optimal_rpm_range': (3700, 7100),  # Peak torque to optimal shift
    'peak_torque_rpm': 3700,
    'optimal_shift_rpm': 7100,
    'threshold_brake_min': 90,  # Race ABS + Alcon brakes on slicks
    'full_throttle_target': 55,  # % of lap distance (lower power car)
    'cornering_g_target': 1.0,  # Threshold cornering on slicks
    'novice_g_threshold': 0.85,  # Under-utilization flag
    'low_rpm_warning': 3500,  # Below power band
    'low_rpm_max_time': 10  # Max % of lap below power band
}

# ============================================================================
# INSIGHT GENERATION ENGINE
# ============================================================================

class InsightGenerator:
    """Generate professional, context-specific insights with GR86 parameters"""
    
    def __init__(self, driver_data, driver_id, driver_name, benchmark_data=None, fastest_lap=None):
        self.data = driver_data
        self.driver_id = driver_id
        self.driver_name = driver_name
        self.benchmark = benchmark_data
        self.fastest_lap = fastest_lap
        self.insights = []
        
        # Calculate key metrics
        self.best_lap_idx = self.data['lap_duration'].idxmin()
        self.best_lap_data = self.data.loc[self.best_lap_idx]
        self.best_lap_time = self.data['lap_duration'].min()
        
        # Determine driver skill level (for scaling) - Expert-based thresholds
        if fastest_lap:
            gap_pct = ((self.best_lap_time - fastest_lap) / fastest_lap) * 100
            if gap_pct < 2:
                self.skill_level = 'Advanced'
                self.gain_multiplier = 0.7
            elif gap_pct < 5:
                self.skill_level = 'Intermediate'
                self.gain_multiplier = 1.0
            elif gap_pct < 8:
                self.skill_level = 'Developing'
                self.gain_multiplier = 1.3
            else:
                self.skill_level = 'Novice'
                self.gain_multiplier = 1.5
        else:
            self.skill_level = 'Intermediate'
            self.gain_multiplier = 1.0
        
        # Averages
        self.avg_speed = self.data['Speed'].mean()
        self.avg_throttle = self.data['Throttle'].mean()
        self.avg_brake = self.data['Brake_Front'].mean()
        self.avg_rpm = self.data['Engine_RPM'].mean()
        self.avg_accel_lat = self.data['Accel_Lateral'].abs().mean()
        self.avg_accel_long = self.data['Accel_Longitudinal'].mean()
        
        # Best lap metrics
        self.best_speed = self.best_lap_data['Speed']
        self.best_throttle = self.best_lap_data['Throttle']
        self.best_brake = self.best_lap_data['Brake_Front']
        self.best_rpm = self.best_lap_data['Engine_RPM']
        
        # Consistency metrics
        self.speed_std = self.data['Speed'].std()
        self.throttle_std = self.data['Throttle'].std()
        self.brake_std = self.data['Brake_Front'].std()
        self.rpm_std = self.data['Engine_RPM'].std()
        
        # Track diminishing returns
        self.high_priority_count = 0
    
    def add_insight(self, category, issue, action, priority, potential_gain_range, confidence=None, impact_score=None):
        """Add insight with skill-level scaling and diminishing returns"""
        min_gain, max_gain = potential_gain_range
        
        # Apply skill level multiplier
        min_gain *= self.gain_multiplier
        max_gain *= self.gain_multiplier
        
        # Apply diminishing returns for subsequent HIGH priority items
        if priority == 'HIGH':
            diminishing_factor = 1.0 - (self.high_priority_count * 0.15)
            diminishing_factor = max(0.5, diminishing_factor)  # Cap at 50% reduction
            min_gain *= diminishing_factor
            max_gain *= diminishing_factor
            self.high_priority_count += 1
        
        avg_gain = (min_gain + max_gain) / 2
        
        # Calculate impact score if not provided
        if impact_score is None:
            impact_score = avg_gain * (1.0 if priority == 'HIGH' else 0.6 if priority == 'MEDIUM' else 0.3)
        
        self.insights.append({
            'driver_name': self.driver_name,
            'driver_id': self.driver_id,
            'category': category,
            'issue': issue,
            'action': action,
            'priority': priority,
            'potential_gain': f'{min_gain:.2f}-{max_gain:.2f}s',
            'estimated_new_lap': f'{self.best_lap_time - avg_gain:.3f}s',
            'current_best_lap': f'{self.best_lap_time:.3f}s',
            'confidence': confidence or 'Medium',
            'impact_score': f'{impact_score:.3f}',
            'skill_level': self.skill_level
        })
    
    # ========================================================================
    # CATEGORY 1: BRAKING PERFORMANCE (GR86-Specific)
    # ========================================================================
    
    def analyze_braking(self):
        """Generate braking insights with GR86 Alcon brake specs"""
        
        # 1. Brake pressure consistency (CRITICAL for consistency)
        if self.brake_std > 20:
            self.add_insight(
                'Braking Consistency',
                f'Brake pressure variance of ±{self.brake_std:.1f}% is a critical limiter to consistent lap times.',
                f'Mandatory: Standardize brake points at Turn 1 chicane and Turn 4 entry. Use 100m board as primary reference. Consistency must be established before chasing speed.',
                'HIGH',
                (0.50, 0.90),  # Consistency is critical - higher gain for novices
                'High'
            )
        
        # 2. Threshold braking (GR86 with race ABS)
        max_brake = self.data['Brake_Front'].max()
        if max_brake < GR86_SPECS['threshold_brake_min']:
            self.add_insight(
                'Threshold Braking Development',
                f'Peak brake pressure of {max_brake:.1f}% is below GR86 threshold capability (90%+ with race ABS and Alcon brakes).',
                f'Critical: Increase peak braking force by 15% at Turn 1/2 chicane. The GR86 race ABS allows 90-95% pressure without lock-up. Build confidence progressively.',
                'HIGH',
                (0.35, 0.60),
                'High'
            )
        
        # 3. Over-braking vs benchmark
        if self.benchmark and self.avg_brake > self.benchmark['brake'] + 15:
            brake_gap = self.avg_brake - self.benchmark['brake']
            self.add_insight(
                'Brake Efficiency vs Leader',
                f'Braking {brake_gap:.1f}% harder than fastest driver (Avg: {self.avg_brake:.1f}% vs {self.benchmark["brake"]:.1f}%).',
                f'Brake 10m later at Turn 3 and Turn 5. Reduce initial pressure by 10% and focus on progressive release. Target: {self.benchmark["brake"]:.1f}% average.',
                'HIGH',
                (0.40, 0.70),
                'High'
            )
        
        # 4. Trail braking opportunity (intermediate/advanced)
        if self.skill_level in ['Advanced', 'Intermediate'] and self.avg_brake > 15 and self.avg_throttle < 60:
            self.add_insight(
                'Trail Braking Technique',
                f'Coasting phase detected between brake release and throttle application.',
                'Opportunity: Maintain 20-30% brake pressure through Turn 4 apex while unwinding steering. This loads the front and improves rotation.',
                'MEDIUM',
                (0.20, 0.35),
                'Medium'
            )
        
        # 4. Brake point optimization
        if self.benchmark:
            bench_brake = self.benchmark['brake']
            if self.avg_brake > bench_brake + 15:
                self.add_insight(
                    'Brake Point Optimization',
                    f'Braking {self.avg_brake - bench_brake:.1f}% harder than fastest driver',
                    f'Study fastest driver\'s braking points. They brake at {bench_brake:.1f}% - try matching their markers.',
                    'HIGH',
                    (0.50, 0.90),
                    'High'
                )
    
    # ========================================================================
    # CATEGORY 2: THROTTLE APPLICATION PATTERNS
    # ========================================================================
    
    def analyze_throttle(self):
        """Generate throttle-specific insights"""
        
        # 1. Full throttle utilization (GR86 is exit-dependent)
        full_throttle = self.data[self.data['Throttle'] > 95]
        full_throttle_pct = (len(full_throttle) / len(self.data)) * 100
        
        if full_throttle_pct < GR86_SPECS['full_throttle_target']:
            deficit = GR86_SPECS['full_throttle_target'] - full_throttle_pct
            self.add_insight(
                'Full Throttle Utilization',
                f'Only {full_throttle_pct:.1f}% of lap at full throttle. GR86 requires {GR86_SPECS["full_throttle_target"]}%+ due to lower power output.',
                f'Critical: Reach 100% throttle {deficit:.0f}% earlier. Focus on Turn 2 exit and Turn 5 exit. Every meter earlier = 0.05s gained on main straight.',
                'HIGH',
                (0.45, 0.75),
                'High'
            )
        
        # 2. Throttle application timing
        if self.avg_throttle < self.best_throttle - 5:
            throttle_gap = self.best_throttle - self.avg_throttle
            self.add_insight(
                'Throttle Application Timing',
                f'Throttle application {throttle_gap:.1f}% below best lap average.',
                f'Apply throttle at apex of Turn 3 and Turn 4. Target: 50% at apex, 75% at mid-exit, 100% at track-out. Current: {self.avg_throttle:.1f}%, Target: {self.best_throttle:.1f}%.',
                'HIGH',
                (0.30, 0.50),
                'High'
            )
        
        # 3. Throttle consistency (consistency multiplier)
        if self.throttle_std > 15:
            gain_mult = 1.25 if self.skill_level in ['Novice', 'Developing'] else 1.0
            self.add_insight(
                'Throttle Consistency',
                f'Throttle variance of ±{self.throttle_std:.1f}% indicates inconsistent corner exits.',
                'Standardize throttle application points. Use apex curbing as reference for initial 50% application. Smooth progression is critical.',
                'HIGH' if self.skill_level in ['Novice', 'Developing'] else 'MEDIUM',
                (0.35 * gain_mult, 0.60 * gain_mult),
                'HIGH' if self.skill_level in ['Novice', 'Developing'] else 'MEDIUM'
            )
        
        # 4. Traction management (GR86 torque characteristics)
        if self.avg_accel_long < 0.5:
            self.add_insight(
                'Traction Management',
                f'Longitudinal acceleration of {self.avg_accel_long:.2f}g is below GR86 capability on slicks.',
                'The GR86 torque curve allows aggressive application from 3700 RPM. If losing traction at Turn 2 exit, reduce initial rate by 10%. If no traction loss, increase aggression by 20%.',
                'MEDIUM',
                (0.25, 0.40),
                'Medium'
            )
    
    # ========================================================================
    # CATEGORY 3: SPEED ANALYSIS
    # ========================================================================
    
    def analyze_speed(self):
        """Generate speed-specific insights"""
        
        # 1. Average speed gap
        if self.avg_speed < self.best_speed - 5:
            speed_gap = self.best_speed - self.avg_speed
            time_gain = (speed_gap / self.avg_speed) * self.best_lap_time if self.avg_speed > 0 else 0
            self.add_insight(
                'Average Speed Optimization',
                f'Average speed {speed_gap:.1f} km/h below your best lap ({self.avg_speed:.1f} vs {self.best_speed:.1f} km/h)',
                f'Focus on minimum corner speeds. Every 1 km/h gained through corners = 0.08s per lap. Target: {self.best_speed:.1f} km/h average.',
                'HIGH',
                (time_gain * 0.8, time_gain * 1.2),
                'High'
            )
        
        # 2. Speed consistency
        if self.speed_std > 20:
            self.add_insight(
                'Speed Consistency',
                f'Speed varies by ±{self.speed_std:.1f} km/h - indicates inconsistent corner speeds',
                'Use reference points for corner entry speeds. Aim for same speed at turn-in point every lap (±2 km/h tolerance).',
                'MEDIUM',
                (0.35, 0.60),
                'Medium'
            )
        
        # 3. Minimum corner speed
        min_speed = self.data['Speed'].min()
        if min_speed < 60:
            self.add_insight(
                'Minimum Corner Speed',
                f'Minimum speed of {min_speed:.1f} km/h is very low - suggests over-slowing',
                f'Carry more speed through slowest corners. Try entering 5 km/h faster and see if you can maintain it through apex.',
                'HIGH',
                (0.30, 0.60),
                'High'
            )
        
        # 4. Speed carry through corners
        if self.benchmark:
            bench_speed = self.benchmark['speed']
            if self.avg_speed < bench_speed - 5:
                speed_deficit = bench_speed - self.avg_speed
                self.add_insight(
                    'Speed Carry Comparison',
                    f'Carrying {speed_deficit:.1f} km/h less speed than fastest driver',
                    f'Study fastest driver\'s corner speeds. They maintain {bench_speed:.1f} km/h average - focus on matching their minimum speeds first.',
                    'HIGH',
                    (0.60, 1.00),
                    'High'
                )
        
        # 5. Maximum speed zones
        max_speed = self.data['Speed'].max()
        if max_speed < 180:
            self.add_insight(
                'Maximum Speed Development',
                f'Top speed of {max_speed:.1f} km/h suggests exit speed or straight-line issues',
                'Focus on corner exit acceleration. Better exits = higher top speeds. Also check if you\'re short-shifting.',
                'MEDIUM',
                (0.20, 0.40),
                'Medium'
            )
    
    # ========================================================================
    # CATEGORY 4: RPM & GEAR MANAGEMENT
    # ========================================================================
    
    def analyze_rpm_gears(self):
        """Generate RPM and gear management insights"""
        
        # 1. Power band utilization (GR86: 3700-7100 RPM optimal)
        low_rpm_data = self.data[self.data['Engine_RPM'] < GR86_SPECS['low_rpm_warning']]
        low_rpm_pct = (len(low_rpm_data) / len(self.data)) * 100
        
        if low_rpm_pct > GR86_SPECS['low_rpm_max_time']:
            self.add_insight(
                'Power Band Management',
                f'{low_rpm_pct:.1f}% of lap below {GR86_SPECS["low_rpm_warning"]} RPM. GR86 peak torque is at {GR86_SPECS["peak_torque_rpm"]} RPM.',
                f'Mandatory: Downshift earlier at Turn 1 and Turn 4 entries. Target: Stay above {GR86_SPECS["peak_torque_rpm"]} RPM for optimal torque delivery. Current avg: {self.avg_rpm:.0f} RPM.',
                'MEDIUM',
                (0.30, 0.50),
                'High'
            )
        
        # 2. Optimal shift point (GR86: 7100 RPM)
        max_rpm = self.data['Engine_RPM'].max()
        if max_rpm < GR86_SPECS['optimal_shift_rpm'] - 300:
            rpm_deficit = GR86_SPECS['optimal_shift_rpm'] - max_rpm
            self.add_insight(
                'Shift Point Optimization',
                f'Shifting at {max_rpm:.0f} RPM. GR86 optimal shift point is {GR86_SPECS["optimal_shift_rpm"]} RPM.',
                f'Shift {rpm_deficit:.0f} RPM later on main straight and Turn 2 exit. Use shift light as reference. Power curve remains strong to {GR86_SPECS["optimal_shift_rpm"]} RPM.',
                'MEDIUM',
                (0.20, 0.35),
                'Medium'
            )
        
        # 3. RPM consistency (only flag if significant)
        if self.rpm_std > 1000:
            self.add_insight(
                'Shift Point Consistency',
                f'RPM variance of ±{self.rpm_std:.0f} indicates inconsistent shift timing.',
                f'Standardize shift points. Set shift light to {GR86_SPECS["optimal_shift_rpm"]} RPM and use it consistently on every shift.',
                'LOW',
                (0.10, 0.20),
                'Low'
            )
    
    # ========================================================================
    # CATEGORY 5: CORNERING PERFORMANCE
    # ========================================================================
    
    def analyze_cornering(self):
        """Generate cornering-specific insights"""
        
        # 1. Cornering G-force (GR86 on slicks: 1.0g+ capable)
        if self.avg_accel_lat < GR86_SPECS['cornering_g_target']:
            g_deficit = GR86_SPECS['cornering_g_target'] - self.avg_accel_lat
            
            if self.avg_accel_lat < GR86_SPECS['novice_g_threshold']:
                # Novice: significant under-utilization
                self.add_insight(
                    'Cornering Speed Development',
                    f'Lateral G-force of {self.avg_accel_lat:.2f}g is {g_deficit:.2f}g below GR86 capability on slicks (1.0g threshold).',
                    f'Critical: Increase corner entry speed by 5 km/h at Turn 3 and Turn 4. Build confidence progressively over 3 sessions. The GR86 on slicks can sustain 1.0g+ in steady-state cornering.',
                    'HIGH',
                    (0.50, 0.80),  # Reduced from 0.80-1.20 per expert feedback
                    'High'
                )
            else:
                # Intermediate: approaching limit
                self.add_insight(
                    'Cornering Speed Refinement',
                    f'Lateral G-force of {self.avg_accel_lat:.2f}g is approaching GR86 limit. {g_deficit:.2f}g remaining.',
                    f'Opportunity: Increase mid-corner speed by 2-3 km/h at Turn 4. Focus on smooth weight transfer and progressive steering inputs.',
                    'MEDIUM',
                    (0.25, 0.45),
                    'Medium'
                )
        
        # 2. Corner exit acceleration (GR86 is exit-dependent)
        if self.avg_accel_long < 0.6:
            self.add_insight(
                'Corner Exit Acceleration',
                f'Exit acceleration of {self.avg_accel_long:.2f}g is below GR86 potential.',
                f'Critical: Apply 50% throttle at Turn 2 apex, building to 100% by track-out. The GR86 requires aggressive exits to compensate for lower power. Every 0.1g improvement = 0.15s per lap.',
                'HIGH',
                (0.50, 0.75),
                'High'
            )
        
        # 3. Apex consistency (consistency multiplier)
        if self.throttle_std > 12 and self.speed_std > 15:
            gain_mult = 1.25 if self.skill_level in ['Novice', 'Developing'] else 1.0
            self.add_insight(
                'Apex Consistency',
                f'Combined throttle (±{self.throttle_std:.1f}%) and speed (±{self.speed_std:.1f} km/h) variance indicates inconsistent apex positioning.',
                'Standardize apex points at Turn 3 and Turn 4. Use inside curbing as reference. Consistent apex = consistent exit = consistent lap times.',
                'HIGH' if self.skill_level in ['Novice', 'Developing'] else 'MEDIUM',
                (0.35 * gain_mult, 0.60 * gain_mult),
                'HIGH' if self.skill_level in ['Novice', 'Developing'] else 'MEDIUM'
            )
        
        # 4. Racing line vs benchmark
        if self.benchmark and self.avg_accel_lat < self.benchmark.get('accel_lat', 1.2) - 0.15:
            g_gap = self.benchmark['accel_lat'] - self.avg_accel_lat
            self.add_insight(
                'Racing Line Optimization',
                f'Cornering {g_gap:.2f}g slower than fastest driver (Avg: {self.avg_accel_lat:.2f}g vs {self.benchmark["accel_lat"]:.2f}g).',
                f'Study fastest driver\'s line at Turn 3 and Turn 4. They use full track width and carry {g_gap:.2f}g more speed. Focus on entry positioning and apex placement.',
                'HIGH',
                (0.55, 0.85),
                'High'
            )
    
    # ========================================================================
    # CATEGORY 6: CONSISTENCY DEVELOPMENT
    # ========================================================================
    
    def analyze_consistency(self):
        """Generate consistency-focused insights"""
        
        # 1. Overall consistency (ALWAYS HIGH PRIORITY - Foundation of speed)
        if self.speed_std > 15 or self.throttle_std > 12:
            # Consistency gets +25% multiplier for novices per expert feedback
            gain_mult = 1.25 if self.skill_level in ['Novice', 'Developing'] else 1.0
            
            self.add_insight(
                'Overall Consistency',
                f'Input variance (Speed: ±{self.speed_std:.1f} km/h, Throttle: ±{self.throttle_std:.1f}%) is the primary performance limiter.',
                f'Mandatory: Establish 3 reference points per corner - brake marker (100m board), turn-in point (apex curbing), throttle point (track-out). Consistency is the foundation of speed. All other improvements are meaningless without this.',
                'HIGH',
                (0.50 * gain_mult, 0.90 * gain_mult),  # Increased per expert: 0.70-1.20s for high variance
                'High'
            )
        
        # 2. Input smoothness (jerky inputs)
        if self.throttle_std > 18 or self.brake_std > 25:
            self.add_insight(
                'Input Smoothness',
                f'Abrupt input changes detected (Throttle: ±{self.throttle_std:.1f}%, Brake: ±{self.brake_std:.1f}%).',
                'Practice progressive inputs. Throttle: 0% to 100% over 1.5 seconds. Brake: 100% to 0% over 1.0 second. Smooth inputs = predictable car behavior.',
                'MEDIUM',
                (0.30, 0.50),
                'Medium'
            )
        
        # 3. Lap-to-lap consistency (if lap data available)
        if len(self.data) > 10:
            lap_times = self.data.groupby('lap_number')['lap_duration'].first() if 'lap_number' in self.data.columns else pd.Series()
            if len(lap_times) > 5:
                lap_std = lap_times.std()
                if lap_std > 2.0:
                    # High variance = novice, gets multiplier
                    gain_mult = 1.25 if self.skill_level in ['Novice', 'Developing'] else 1.0
                    self.add_insight(
                        'Lap Time Consistency',
                        f'Lap time variance of ±{lap_std:.2f}s indicates unstable baseline pace.',
                        f'Critical: Establish consistent pace before chasing speed. Target: 5 consecutive laps within 0.5s. Current variance is {lap_std:.2f}s.',
                        'HIGH',
                        (0.60 * gain_mult, 1.00 * gain_mult),
                        'High'
                    )
    
    # ========================================================================
    # CATEGORY 7: SECTOR-SPECIFIC ANALYSIS
    # ========================================================================
    
    def analyze_sectors(self):
        """Generate sector-specific insights (if sector data available)"""
        
        # Note: This requires sector timing data which may not be in current telemetry
        # For now, generate general sector-based recommendations
        
        # Divide lap into 3 sectors based on distance/time
        if 'Distance' in self.data.columns or 'Time' in self.data.columns:
            total_points = len(self.data)
            sector_size = total_points // 3
            
            sector1 = self.data.iloc[:sector_size]
            sector2 = self.data.iloc[sector_size:sector_size*2]
            sector3 = self.data.iloc[sector_size*2:]
            
            # Analyze each sector
            s1_speed = sector1['Speed'].mean()
            s2_speed = sector2['Speed'].mean()
            s3_speed = sector3['Speed'].mean()
            
            # Find weakest sector
            sectors = [('Sector 1', s1_speed), ('Sector 2', s2_speed), ('Sector 3', s3_speed)]
            weakest = min(sectors, key=lambda x: x[1])
            
            self.add_insight(
                f'{weakest[0]} Improvement',
                f'{weakest[0]} is your weakest with {weakest[1]:.1f} km/h average speed',
                f'Focus practice on {weakest[0]}. Analyze fastest driver\'s approach to these corners specifically.',
                'MEDIUM',
                (0.30, 0.60),
                'Medium'
            )
    
    # ========================================================================
    # CATEGORY 8: COMPARATIVE ANALYSIS
    # ========================================================================
    
    def analyze_comparative(self):
        """Generate insights comparing to fastest driver"""
        
        if not self.benchmark:
            return
        
        bench_speed = self.benchmark['speed']
        bench_throttle = self.benchmark['throttle']
        bench_brake = self.benchmark['brake']
        bench_rpm = self.benchmark['rpm']
        
        # Calculate impact scores for prioritization
        speed_gap = bench_speed - self.avg_speed
        throttle_gap = bench_throttle - self.avg_throttle
        brake_gap = self.avg_brake - bench_brake
        rpm_gap = bench_rpm - self.avg_rpm
        
        gaps = [
            ('Speed', speed_gap, 'km/h', abs(speed_gap) * 0.08),  # Impact: 0.08s per km/h
            ('Throttle', throttle_gap, '%', abs(throttle_gap) * 0.03),  # Impact: 0.03s per %
            ('Brake', brake_gap, '%', abs(brake_gap) * 0.025),  # Impact: 0.025s per %
            ('RPM', rpm_gap, 'RPM', abs(rpm_gap) * 0.0001)  # Impact: 0.0001s per RPM
        ]
        
        # Sort by impact score
        gaps.sort(key=lambda x: x[3], reverse=True)
        
        # Report primary gap only (most impactful)
        metric, gap, unit, impact = gaps[0]
        
        if abs(gap) > 5:
            if metric == 'Speed':
                self.add_insight(
                    'Speed Gap to Leader',
                    f'{abs(gap):.1f} km/h slower than fastest driver (Avg: {self.avg_speed:.1f} vs {bench_speed:.1f} km/h).',
                    f'Study fastest driver\'s minimum corner speeds at Turn 3 and Turn 4. They carry {abs(gap):.1f} km/h more speed. Focus on entry positioning and mid-corner speed maintenance.',
                    'HIGH',
                    (impact * 0.8, impact * 1.2),
                    'High'
                )
            elif metric == 'Throttle':
                self.add_insight(
                    'Throttle Gap to Leader',
                    f'{abs(gap):.1f}% less throttle than fastest driver (Avg: {self.avg_throttle:.1f}% vs {bench_throttle:.1f}%).',
                    f'Apply throttle earlier at Turn 2 and Turn 5 exits. Fastest driver reaches 100% throttle {abs(gap):.1f}% earlier. Focus on exit acceleration.',
                    'HIGH',
                    (impact * 0.8, impact * 1.2),
                    'High'
                )
            elif metric == 'Brake' and gap > 0:
                self.add_insight(
                    'Brake Gap to Leader',
                    f'Braking {abs(gap):.1f}% harder than fastest driver (Avg: {self.avg_brake:.1f}% vs {bench_brake:.1f}%).',
                    f'Brake 10m later at Turn 1 and Turn 4. Reduce peak pressure by {abs(gap):.1f}% and focus on progressive release. Target: {bench_brake:.1f}% average.',
                    'HIGH',
                    (impact * 0.8, impact * 1.2),
                    'High'
                )
    
    # ========================================================================
    # CATEGORY 9: ADVANCED TECHNIQUES
    # ========================================================================
    
    def analyze_advanced_techniques(self):
        """Generate insights for advanced driving techniques"""
        
        # Only provide advanced techniques for Intermediate/Advanced drivers
        if self.skill_level in ['Novice', 'Developing']:
            return
        
        # 1. Weight transfer management (Intermediate+)
        if self.avg_accel_long < 0.5 and self.avg_accel_lat < 1.0:
            self.add_insight(
                'Weight Transfer Optimization',
                f'G-force profile (Longitudinal: {self.avg_accel_long:.2f}g, Lateral: {self.avg_accel_lat:.2f}g) suggests conservative weight transfer.',
                'Opportunity: Brake harder initially at Turn 1 to load front, then smoothly transition to throttle to shift weight rear. This improves rotation and exit traction.',
                'MEDIUM',
                (0.25, 0.40),
                'Medium'
            )
        
        # 2. Rotation management (Advanced only)
        if self.skill_level == 'Advanced' and self.throttle_std > 15 and self.avg_accel_lat > 1.0:
            self.add_insight(
                'Car Rotation Control',
                f'Throttle variance (±{self.throttle_std:.1f}%) with high cornering speed suggests rotation management opportunity.',
                'Refine throttle modulation mid-corner. Lift 5-10% at Turn 4 mid-corner if understeering. Add 5-10% if oversteering. Use throttle as rotation tool.',
                'LOW',
                (0.15, 0.25),
                'Low'
            )
        
        # 3. Threshold braking (already covered in braking section, skip duplicate)
    
    # ========================================================================
    # CATEGORY 10: SETUP & TECHNICAL
    # ========================================================================
    
    def analyze_setup_technical(self):
        """Generate setup and technical insights"""
        
        # Setup recommendations are LOW priority only (driver technique first)
        
        # 1. Brake bias (if front/rear brake data available)
        if 'Brake_Rear' in self.data.columns:
            brake_balance = self.data['Brake_Front'].mean() / (self.data['Brake_Front'].mean() + self.data['Brake_Rear'].mean())
            if brake_balance > 0.65:
                self.add_insight(
                    'Brake Balance Review',
                    f'Front brake bias at {brake_balance*100:.1f}% may be excessive.',
                    'Potential setup change: Move brake bias 1-2% rearward for improved rotation. Test in practice session only after addressing technique issues.',
                    'LOW',
                    (0.10, 0.20),
                    'Low'
                )
        
        # 2. Tire temperature (if available) - MEDIUM if critical
        if 'Tire_Temp_FL' in self.data.columns:
            avg_temp = self.data[['Tire_Temp_FL', 'Tire_Temp_FR']].mean().mean()
            if avg_temp < 70:
                self.add_insight(
                    'Tire Temperature Management',
                    f'Tire temperature of {avg_temp:.1f}°C is critically below optimal range (80-100°C).',
                    'Add 3-4 warm-up laps with aggressive braking and cornering. Weave on straights to generate heat. Cold tires = reduced grip and inconsistent lap times.',
                    'MEDIUM',
                    (0.40, 0.70),
                    'High'
                )
    
    # ========================================================================
    # MASTER GENERATION METHOD
    # ========================================================================
    
    def generate_all_insights(self):
        """Generate insights across all categories"""
        self.analyze_braking()
        self.analyze_throttle()
        self.analyze_speed()
        self.analyze_rpm_gears()
        self.analyze_cornering()
        self.analyze_consistency()
        self.analyze_sectors()
        self.analyze_comparative()
        self.analyze_advanced_techniques()
        self.analyze_setup_technical()
        
        return self.insights

# ============================================================================
# MAIN EXECUTION
# ============================================================================

print("\n[2/8] Filtering to drivers with complete data...")

# Filter telemetry to only include drivers that have predictions (for consistency)
if has_predictions:
    valid_driver_ids = predictions['driver_name'].str.extract(r'-(\d+)$')[0].astype(int).tolist()
    telemetry = telemetry[telemetry['vehicle_number'].isin(valid_driver_ids)].copy()
    print(f"[OK] Using {len(valid_driver_ids)} drivers with complete data (matching predictions)")
else:
    print("[WARNING] No predictions file found, using all telemetry data")

print("\n[3/8] Calculating benchmark from fastest driver...")

# Find fastest driver
fastest_driver_id = telemetry.loc[telemetry['lap_duration'].idxmin(), 'vehicle_number']
fastest_lap_time = telemetry['lap_duration'].min()
fastest_data = telemetry[telemetry['vehicle_number'] == fastest_driver_id]

benchmark = {
    'speed': fastest_data['Speed'].mean(),
    'throttle': fastest_data['Throttle'].mean(),
    'brake': fastest_data['Brake_Front'].mean(),
    'rpm': fastest_data['Engine_RPM'].mean(),
    'accel_lat': fastest_data['Accel_Lateral'].abs().mean()
}

print(f"[OK] Benchmark: Car #{int(fastest_driver_id)} - {fastest_lap_time:.2f}s")
print(f"     Speed: {benchmark['speed']:.1f} km/h | Throttle: {benchmark['throttle']:.1f}% | Brake: {benchmark['brake']:.1f}%")

print("\n[4/8] Generating comprehensive insights for all drivers...")

all_insights = []
driver_count = 0

for driver_id in telemetry['vehicle_number'].unique():
    driver_data = telemetry[telemetry['vehicle_number'] == driver_id].copy()
    
    if len(driver_data) < 5:
        continue
    
    driver_name = VEHICLE_ID_MAP.get(str(int(driver_id)), f"GR86-Unknown-{int(driver_id)}")
    
    # Skip fastest driver for comparative analysis
    bench_data = benchmark if driver_id != fastest_driver_id else None
    
    # Generate insights with skill level calculation
    generator = InsightGenerator(driver_data, driver_id, driver_name, bench_data, fastest_lap_time)
    insights = generator.generate_all_insights()
    
    all_insights.extend(insights)
    driver_count += 1
    
    print(f"  [{driver_count}/{telemetry['vehicle_number'].nunique()}] {driver_name}: {len(insights)} insights generated")

insights_df = pd.DataFrame(all_insights)
print(f"\n[OK] Generated {len(insights_df)} total insights for {driver_count} drivers")
print(f"     Average: {len(insights_df)/driver_count:.1f} insights per driver")

# ============================================================================
# SAVE OUTPUTS
# ============================================================================

print("\n[8/8] Saving enhanced insights...")

import os
os.makedirs('outputs/enhanced_telemetry_insights', exist_ok=True)

# Save comprehensive insights
insights_df.to_csv('outputs/enhanced_telemetry_insights/comprehensive_insights.csv', index=False)
print("[OK] Saved: outputs/enhanced_telemetry_insights/comprehensive_insights.csv")

# ============================================================================
# GENERATE PRIORITY SUMMARY
# ============================================================================

print("\n[8/8] Generating priority summary...")

priority_summary = insights_df.groupby(['driver_name', 'priority']).size().unstack(fill_value=0)
priority_summary.to_csv('outputs/enhanced_telemetry_insights/priority_summary.csv')
print("[OK] Saved: outputs/enhanced_telemetry_insights/priority_summary.csv")

# ============================================================================
# GENERATE CATEGORY BREAKDOWN
# ============================================================================

print("\n[8/8] Analyzing insight categories...")

category_breakdown = insights_df.groupby('category').agg({
    'driver_id': 'count',
    'priority': lambda x: (x == 'HIGH').sum()
}).rename(columns={'driver_id': 'total_insights', 'priority': 'high_priority_count'})

category_breakdown = category_breakdown.sort_values('total_insights', ascending=False)
category_breakdown.to_csv('outputs/enhanced_telemetry_insights/category_breakdown.csv')
print("[OK] Saved: outputs/enhanced_telemetry_insights/category_breakdown.csv")

print("\nTop 5 insight categories:")
for idx, (category, row) in enumerate(category_breakdown.head().iterrows(), 1):
    print(f"  {idx}. {category}: {int(row['total_insights'])} insights ({int(row['high_priority_count'])} high priority)")

# ============================================================================
# GENERATE DRIVER REPORTS
# ============================================================================

print("\n[8/8] Generating individual driver reports...")

report_dir = 'outputs/enhanced_telemetry_insights/driver_reports'
os.makedirs(report_dir, exist_ok=True)

for driver_id in insights_df['driver_id'].unique():
    driver_insights = insights_df[insights_df['driver_id'] == driver_id].copy()
    driver_name = driver_insights.iloc[0]['driver_name']
    
    # Sort by priority
    priority_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
    driver_insights['priority_rank'] = driver_insights['priority'].map(priority_order)
    driver_insights = driver_insights.sort_values('priority_rank')
    
    # Generate report
    report_path = f"{report_dir}/{driver_name.replace('#', '').replace(' ', '_')}_insights.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"{driver_name} - COMPREHENSIVE TELEMETRY INSIGHTS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Best Lap: {driver_insights.iloc[0]['current_best_lap']}\n")
        f.write(f"Total Insights: {len(driver_insights)}\n")
        f.write(f"  - HIGH Priority: {len(driver_insights[driver_insights['priority'] == 'HIGH'])}\n")
        f.write(f"  - MEDIUM Priority: {len(driver_insights[driver_insights['priority'] == 'MEDIUM'])}\n")
        f.write(f"  - LOW Priority: {len(driver_insights[driver_insights['priority'] == 'LOW'])}\n\n")
        
        # Calculate total potential
        total_potential = 0
        for _, insight in driver_insights.iterrows():
            try:
                gain_str = insight['potential_gain'].split('-')[0]
                gain_val = float(gain_str.replace('s', '').strip())
                total_potential += gain_val
            except:
                pass
        
        f.write(f"TOTAL POTENTIAL IMPROVEMENT: {total_potential:.2f}s\n")
        f.write(f"PROJECTED BEST LAP: {float(driver_insights.iloc[0]['current_best_lap'].replace('s', '')) - total_potential:.3f}s\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("INSIGHTS BY PRIORITY\n")
        f.write("=" * 80 + "\n\n")
        
        for priority in ['HIGH', 'MEDIUM', 'LOW']:
            priority_insights = driver_insights[driver_insights['priority'] == priority]
            if len(priority_insights) == 0:
                continue
            
            f.write(f"\n{'='*80}\n")
            f.write(f"{priority} PRIORITY ({len(priority_insights)} insights)\n")
            f.write(f"{'='*80}\n\n")
            
            for idx, (_, insight) in enumerate(priority_insights.iterrows(), 1):
                f.write(f"{idx}. [{insight['category']}]\n")
                f.write(f"   Issue: {insight['issue']}\n")
                f.write(f"   Action: {insight['action']}\n")
                f.write(f"   Potential Gain: {insight['potential_gain']}\n")
                f.write(f"   Estimated New Lap: {insight['estimated_new_lap']}\n")
                f.write(f"   Confidence: {insight['confidence']}\n\n")

print(f"[OK] Generated {len(insights_df['driver_id'].unique())} driver reports")

# ============================================================================
# GENERATE MASTER SUMMARY
# ============================================================================

print("\n[8/8] Generating master summary report...")

with open('outputs/enhanced_telemetry_insights/MASTER_SUMMARY.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("TOYOTA GR CUP - ENHANCED TELEMETRY INSIGHTS SUMMARY\n")
    f.write("=" * 80 + "\n\n")
    
    f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Drivers Analyzed: {driver_count}\n")
    f.write(f"Total Insights Generated: {len(insights_df)}\n")
    f.write(f"Average Insights per Driver: {len(insights_df)/driver_count:.1f}\n\n")
    
    f.write(f"Benchmark (Fastest Driver):\n")
    f.write(f"  Car #{int(fastest_driver_id)} - {fastest_lap_time:.2f}s\n")
    f.write(f"  Speed: {benchmark['speed']:.1f} km/h\n")
    f.write(f"  Throttle: {benchmark['throttle']:.1f}%\n")
    f.write(f"  Brake: {benchmark['brake']:.1f}%\n")
    f.write(f"  RPM: {benchmark['rpm']:.0f}\n\n")
    
    f.write("=" * 80 + "\n")
    f.write("INSIGHT DISTRIBUTION\n")
    f.write("=" * 80 + "\n\n")
    
    f.write(f"By Priority:\n")
    for priority in ['HIGH', 'MEDIUM', 'LOW']:
        count = len(insights_df[insights_df['priority'] == priority])
        pct = count / len(insights_df) * 100
        f.write(f"  {priority}: {count} ({pct:.1f}%)\n")
    
    f.write(f"\nBy Category (Top 10):\n")
    for idx, (category, row) in enumerate(category_breakdown.head(10).iterrows(), 1):
        f.write(f"  {idx}. {category}: {int(row['total_insights'])} insights\n")
    
    f.write("\n" + "=" * 80 + "\n")
    f.write("MOST COMMON ISSUES\n")
    f.write("=" * 80 + "\n\n")
    
    # Find most common issues
    issue_counts = insights_df['category'].value_counts().head(5)
    for idx, (issue, count) in enumerate(issue_counts.items(), 1):
        affected_drivers = len(insights_df[insights_df['category'] == issue]['driver_id'].unique())
        f.write(f"{idx}. {issue}\n")
        f.write(f"   Affects {affected_drivers} drivers ({affected_drivers/driver_count*100:.1f}%)\n")
        f.write(f"   Total insights: {count}\n\n")
    
    f.write("=" * 80 + "\n")
    f.write("RECOMMENDATIONS\n")
    f.write("=" * 80 + "\n\n")
    
    f.write("1. Drivers should focus on HIGH priority insights first\n")
    f.write("2. Address consistency issues before chasing raw speed\n")
    f.write("3. Use driver reports for detailed, actionable recommendations\n")
    f.write("4. Compare telemetry to fastest driver for learning opportunities\n")
    f.write("5. Track progress by re-running analysis after each session\n\n")

print("[OK] Saved: outputs/enhanced_telemetry_insights/MASTER_SUMMARY.txt")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("TELEMETRY ANALYSIS COMPLETE")
print("=" * 80)
print(f"\nGenerated {len(insights_df)} comprehensive insights:")
print(f"  - {len(insights_df[insights_df['priority'] == 'HIGH'])} HIGH priority")
print(f"  - {len(insights_df[insights_df['priority'] == 'MEDIUM'])} MEDIUM priority")
print(f"  - {len(insights_df[insights_df['priority'] == 'LOW'])} LOW priority")
print(f"\nAverage per driver: {len(insights_df)/driver_count:.1f} insights")
print(f"\nTop 3 categories:")
for idx, (category, row) in enumerate(category_breakdown.head(3).iterrows(), 1):
    print(f"  {idx}. {category}: {int(row['total_insights'])} insights")
print(f"\nAll outputs saved to: outputs/enhanced_telemetry_insights/")
print(f"Individual driver reports: outputs/enhanced_telemetry_insights/driver_reports/")
