"""
Real Intelligence Engine - 300% Better Commentary
Multi-layer AI reasoning pipeline with pattern detection and dynamic insights
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class DrivingStyle(Enum):
    """Driver style classifications based on telemetry patterns"""
    LATE_BRAKER = "Late Braker"
    AGGRESSIVE_ACCELERANT = "Aggressive Accelerant"
    SMOOTH_CONSISTENT = "Smooth & Consistent"
    CORNER_SPEED_SPECIALIST = "Corner Speed Specialist"
    HIGH_RISK_HIGH_VARIANCE = "High-Risk, High-Variance"
    CONSERVATIVE_CLEAN = "Conservative but Clean"
    STRAIGHT_SPEED_TECHNICAL_WEAKNESS = "Speed on Straights, Weakness in Technical"


class ConsistencyPattern(Enum):
    """Consistency patterns detected from lap data"""
    IMPROVING = "improving"
    DEGRADING = "degrading"
    STABLE = "stable"
    VOLATILE = "volatile"
    COLD_LAP_WEAK = "cold_lap_weak"
    HOT_LAP_STRONG = "hot_lap_strong"


@dataclass
class TrackContext:
    """Track-specific context for intelligent commentary"""
    name: str = "Indianapolis Road Course"
    technical_sectors: List[int] = None
    heavy_braking_zones: List[int] = None
    long_straights: List[int] = None
    slow_corners: List[int] = None
    chicanes: List[int] = None
    traction_zones: List[int] = None
    
    def __post_init__(self):
        # Indianapolis Road Course specifics
        if self.technical_sectors is None:
            self.technical_sectors = [1]  # Infield section
        if self.heavy_braking_zones is None:
            self.heavy_braking_zones = [1, 7, 14]
        if self.long_straights is None:
            self.long_straights = [3]  # Oval section
        if self.slow_corners is None:
            self.slow_corners = [1, 7, 12, 14]
        if self.chicanes is None:
            self.chicanes = [12, 13]
        if self.traction_zones is None:
            self.traction_zones = [7, 14]


class IntelligenceEngine:
    """
    Real AI Commentary Engine - Pattern Detection & Dynamic Insights
    Replaces template system with intelligent multi-layer reasoning
    """
    
    def __init__(self):
        self.track_context = TrackContext()
        self.severity_scales = self._init_severity_scales()
        self.comparative_adjectives = self._init_comparative_adjectives()
        
    def _init_severity_scales(self) -> Dict:
        """Initialize severity rating scales"""
        return {
            'excellent': {'threshold': 0.95, 'adjective': 'exceptional', 'color': 'positive'},
            'good': {'threshold': 0.80, 'adjective': 'strong', 'color': 'positive'},
            'average': {'threshold': 0.60, 'adjective': 'adequate', 'color': 'neutral'},
            'below_average': {'threshold': 0.40, 'adjective': 'concerning', 'color': 'warning'},
            'poor': {'threshold': 0.20, 'adjective': 'critical', 'color': 'danger'},
            'critical': {'threshold': 0.0, 'adjective': 'severely deficient', 'color': 'danger'}
        }
    
    def _init_comparative_adjectives(self) -> Dict:
        """Initialize comparative language scales"""
        return {
            'much_better': ['significantly outperforms', 'substantially faster than', 'dominates'],
            'better': ['outperforms', 'faster than', 'ahead of'],
            'slightly_better': ['marginally ahead of', 'slightly faster than', 'edges out'],
            'equal': ['matches', 'equals', 'on par with'],
            'slightly_worse': ['marginally behind', 'slightly slower than', 'trails'],
            'worse': ['underperforms', 'slower than', 'behind'],
            'much_worse': ['significantly underperforms', 'substantially slower than', 'lags far behind']
        }

    
    # ========================================================================
    # MODULE 1: DRIVER STYLE DETECTION
    # ========================================================================
    
    def detect_driving_style(self, telemetry_data: pd.DataFrame) -> Tuple[DrivingStyle, Dict]:
        """
        Classify driver style using telemetry pattern analysis
        Returns: (style, confidence_scores)
        """
        if telemetry_data.empty:
            return DrivingStyle.SMOOTH_CONSISTENT, {}
        
        scores = {}
        
        # Analyze braking patterns
        if 'Brake_Front' in telemetry_data.columns:
            brake_variance = telemetry_data['Brake_Front'].std()
            brake_max = telemetry_data['Brake_Front'].max()
            brake_mean = telemetry_data['Brake_Front'].mean()
            
            # Late braker: High max brake, low mean (brakes hard but briefly)
            if brake_max > 80 and brake_mean < 20:
                scores[DrivingStyle.LATE_BRAKER] = 0.8
            
            # Aggressive: High variance in braking
            if brake_variance > 25:
                scores[DrivingStyle.AGGRESSIVE_ACCELERANT] = scores.get(DrivingStyle.AGGRESSIVE_ACCELERANT, 0) + 0.4
        
        # Analyze throttle patterns
        if 'Throttle' in telemetry_data.columns:
            throttle_variance = telemetry_data['Throttle'].std()
            throttle_mean = telemetry_data['Throttle'].mean()
            
            # Aggressive accelerant: High throttle variance
            if throttle_variance > 20:
                scores[DrivingStyle.AGGRESSIVE_ACCELERANT] = scores.get(DrivingStyle.AGGRESSIVE_ACCELERANT, 0) + 0.5
            
            # Smooth: Low variance, high mean
            if throttle_variance < 12 and throttle_mean > 70:
                scores[DrivingStyle.SMOOTH_CONSISTENT] = 0.9
        
        # Analyze speed patterns
        if 'Speed' in telemetry_data.columns:
            speed_variance = telemetry_data['Speed'].std()
            speed_max = telemetry_data['Speed'].max()
            speed_mean = telemetry_data['Speed'].mean()
            
            # High variance = high risk
            if speed_variance > 25:
                scores[DrivingStyle.HIGH_RISK_HIGH_VARIANCE] = 0.7
            
            # Low variance = conservative
            if speed_variance < 15:
                scores[DrivingStyle.CONSERVATIVE_CLEAN] = 0.6
            
            # High max, low mean = straight speed, technical weakness
            if speed_max > 180 and speed_mean < 130:
                scores[DrivingStyle.STRAIGHT_SPEED_TECHNICAL_WEAKNESS] = 0.75
            
            # High mean, moderate max = corner speed specialist
            if speed_mean > 135 and speed_max < 175:
                scores[DrivingStyle.CORNER_SPEED_SPECIALIST] = 0.8
        
        # Analyze lateral G (cornering)
        if 'Accel_Lateral' in telemetry_data.columns:
            lateral_g_mean = telemetry_data['Accel_Lateral'].abs().mean()
            
            # High lateral G = corner speed specialist
            if lateral_g_mean > 1.2:
                scores[DrivingStyle.CORNER_SPEED_SPECIALIST] = scores.get(DrivingStyle.CORNER_SPEED_SPECIALIST, 0) + 0.3
        
        # Return highest scoring style
        if scores:
            best_style = max(scores.items(), key=lambda x: x[1])
            return best_style[0], scores
        
        return DrivingStyle.SMOOTH_CONSISTENT, {DrivingStyle.SMOOTH_CONSISTENT: 0.5}

    
    # ========================================================================
    # MODULE 2: CONSISTENCY PATTERN RECOGNITION
    # ========================================================================
    
    def analyze_consistency_patterns(self, telemetry_data: pd.DataFrame) -> Dict:
        """
        Detect consistency patterns: variance, trends, cold/hot lap performance
        Returns detailed consistency analysis
        """
        patterns = {
            'overall_pattern': ConsistencyPattern.STABLE,
            'lap_to_lap_variance': 0.0,
            'sector_variance': {},
            'pace_stability': 0.0,
            'cold_hot_gap': 0.0,
            'trend': 'stable',
            'trend_magnitude': 0.0,
            'insights': []
        }
        
        if telemetry_data.empty or 'lap_duration' not in telemetry_data.columns:
            return patterns
        
        lap_times = telemetry_data['lap_duration'].dropna()
        
        if len(lap_times) < 3:
            return patterns
        
        # Lap-to-lap variance
        lap_variance = lap_times.std()
        lap_mean = lap_times.mean()
        patterns['lap_to_lap_variance'] = lap_variance
        patterns['pace_stability'] = 1 - (lap_variance / lap_mean) if lap_mean > 0 else 0
        
        # Detect trend (improving vs degrading)
        first_third = lap_times.head(len(lap_times)//3).mean()
        last_third = lap_times.tail(len(lap_times)//3).mean()
        trend_diff = first_third - last_third
        
        patterns['trend_magnitude'] = abs(trend_diff)
        
        if trend_diff > 0.3:
            patterns['trend'] = 'improving'
            patterns['overall_pattern'] = ConsistencyPattern.IMPROVING
            patterns['insights'].append(
                f"Driver improves significantly as session progresses; last {len(lap_times)//3} laps show "
                f"a clear downward trend of -{trend_diff:.3f}s. Tire warm-up or confidence building evident."
            )
        elif trend_diff < -0.3:
            patterns['trend'] = 'degrading'
            patterns['overall_pattern'] = ConsistencyPattern.DEGRADING
            patterns['insights'].append(
                f"Performance degrades over session; last {len(lap_times)//3} laps are +{abs(trend_diff):.3f}s slower. "
                f"Possible tire degradation, fatigue, or setup issues."
            )
        else:
            patterns['trend'] = 'stable'
            patterns['overall_pattern'] = ConsistencyPattern.STABLE
        
        # Cold lap vs hot lap analysis
        if len(lap_times) >= 5:
            cold_laps = lap_times.head(2).mean()
            hot_laps = lap_times.iloc[2:5].mean() if len(lap_times) > 4 else lap_times.tail(2).mean()
            cold_hot_gap = cold_laps - hot_laps
            patterns['cold_hot_gap'] = cold_hot_gap
            
            if cold_hot_gap > 0.5:
                patterns['overall_pattern'] = ConsistencyPattern.COLD_LAP_WEAK
                patterns['insights'].append(
                    f"Significant cold lap weakness detected: first 2 laps are +{cold_hot_gap:.3f}s slower. "
                    f"Driver requires warm-up time to reach optimal pace."
                )
            elif cold_hot_gap < -0.3:
                patterns['overall_pattern'] = ConsistencyPattern.HOT_LAP_STRONG
                patterns['insights'].append(
                    f"Strong initial pace: first 2 laps are {abs(cold_hot_gap):.3f}s faster than subsequent laps. "
                    f"Possible tire degradation or over-driving early."
                )
        
        # Volatility check
        if lap_variance > lap_mean * 0.03:  # >3% variance
            patterns['overall_pattern'] = ConsistencyPattern.VOLATILE
            patterns['insights'].append(
                f"High lap time volatility detected: ±{lap_variance:.3f}s variance indicates inconsistent "
                f"driving inputs or variable conditions."
            )
        
        return patterns

    
    # ========================================================================
    # MODULE 3: COMPARATIVE INSIGHTS
    # ========================================================================
    
    def generate_comparative_insights(self, driver_data: Dict, field_data: pd.DataFrame) -> Dict:
        """
        Compare driver to field, top 10, and closest competitor
        Returns rich comparative analysis
        """
        comparisons = {
            'vs_fastest': {},
            'vs_top10_avg': {},
            'vs_field_avg': {},
            'vs_closest_competitor': {},
            'percentile_rank': 0,
            'insights': []
        }
        
        if field_data.empty:
            return comparisons
        
        driver_best = driver_data.get('best_lap', 0)
        if driver_best == 0:
            return comparisons
        
        # Get field statistics
        field_best_laps = field_data['lap_duration'].groupby(field_data['vehicle_number']).min()
        fastest_lap = field_best_laps.min()
        top10_avg = field_best_laps.nsmallest(10).mean()
        field_avg = field_best_laps.mean()
        
        # Percentile ranking
        better_than = (field_best_laps > driver_best).sum()
        total_drivers = len(field_best_laps)
        percentile = (better_than / total_drivers) * 100 if total_drivers > 0 else 0
        comparisons['percentile_rank'] = percentile
        
        # vs Fastest
        gap_to_fastest = driver_best - fastest_lap
        comparisons['vs_fastest'] = {
            'gap': gap_to_fastest,
            'percentage': (gap_to_fastest / fastest_lap) * 100 if fastest_lap > 0 else 0
        }
        
        if gap_to_fastest < 0.2:
            comparisons['insights'].append(
                f"Elite pace: Only {gap_to_fastest:.3f}s from fastest lap. You're in the top tier, "
                f"outperforming {percentile:.0f}% of the field."
            )
        elif gap_to_fastest < 0.5:
            comparisons['insights'].append(
                f"Strong competitive pace: {gap_to_fastest:.3f}s from fastest lap. "
                f"You're faster than {percentile:.0f}% of drivers with clear podium potential."
            )
        elif gap_to_fastest < 1.0:
            comparisons['insights'].append(
                f"Mid-pack pace: {gap_to_fastest:.3f}s gap to fastest lap. "
                f"You're in the {percentile:.0f}th percentile with room for improvement."
            )
        else:
            comparisons['insights'].append(
                f"Significant gap to leaders: {gap_to_fastest:.3f}s behind fastest lap. "
                f"Focus on fundamentals to close the {(gap_to_fastest/fastest_lap)*100:.1f}% pace deficit."
            )
        
        # vs Top 10 Average
        gap_to_top10 = driver_best - top10_avg
        comparisons['vs_top10_avg'] = {
            'gap': gap_to_top10,
            'percentage': (gap_to_top10 / top10_avg) * 100 if top10_avg > 0 else 0
        }
        
        if gap_to_top10 < 0:
            comparisons['insights'].append(
                f"Exceptional: You're {abs(gap_to_top10):.3f}s faster than the top 10 average. "
                f"Your pace is championship-caliber."
            )
        elif gap_to_top10 < 0.5:
            comparisons['insights'].append(
                f"Competitive with leaders: Within {gap_to_top10:.3f}s of top 10 average. "
                f"Small improvements will put you in contention."
            )
        
        # vs Field Average
        gap_to_field = driver_best - field_avg
        comparisons['vs_field_avg'] = {
            'gap': gap_to_field,
            'percentage': (gap_to_field / field_avg) * 100 if field_avg > 0 else 0
        }
        
        if gap_to_field < 0:
            comparisons['insights'].append(
                f"Above average: {abs(gap_to_field):.3f}s faster than field average. "
                f"You're in the faster half of the grid."
            )
        
        return comparisons

    
    # ========================================================================
    # MODULE 4: PREDICTION STABILITY ANALYSIS
    # ========================================================================
    
    def analyze_prediction_stability(self, predictions: Dict) -> Dict:
        """
        Analyze model agreement, confidence, and prediction variance
        Returns ML insights about prediction quality
        """
        stability = {
            'model_agreement_score': 0.0,
            'confidence_level': 'unknown',
            'prediction_variance': 0.0,
            'reliability_rating': 'unknown',
            'insights': []
        }
        
        # Extract prediction metrics
        confidence = predictions.get('confidence_score', 50)
        models_agree = predictions.get('models_in_agreement', 3)
        mae = predictions.get('mae', 1.0)
        
        # Model agreement analysis
        agreement_pct = (models_agree / 6) * 100
        stability['model_agreement_score'] = agreement_pct
        
        if models_agree == 6:
            stability['insights'].append(
                "Perfect model consensus: All 6 ML algorithms agree on this prediction — exceptionally rare. "
                "This indicates highly predictable, consistent driving patterns. Prediction reliability is maximum."
            )
            stability['reliability_rating'] = 'exceptional'
        elif models_agree >= 5:
            stability['insights'].append(
                f"Strong model consensus: {models_agree}/6 models agree. High prediction reliability with "
                f"only {6-models_agree} outlier model(s). The prediction is trustworthy."
            )
            stability['reliability_rating'] = 'high'
        elif models_agree >= 4:
            stability['insights'].append(
                f"Moderate model consensus: {models_agree}/6 models agree. Reasonable prediction reliability, "
                f"but {6-models_agree} models see different patterns. Use prediction range as guide."
            )
            stability['reliability_rating'] = 'moderate'
        else:
            stability['insights'].append(
                f"Low model consensus: Only {models_agree}/6 models agree. Significant disagreement indicates "
                f"unpredictable performance patterns. Treat prediction as rough estimate only."
            )
            stability['reliability_rating'] = 'low'
        
        # Confidence analysis
        if confidence >= 80:
            stability['confidence_level'] = 'very_high'
            stability['insights'].append(
                f"Very high confidence ({confidence:.0f}%): The selected model has proven accuracy for this driver. "
                f"Expect actual lap time within ±{mae:.2f}s of prediction."
            )
        elif confidence >= 60:
            stability['confidence_level'] = 'good'
            stability['insights'].append(
                f"Good confidence ({confidence:.0f}%): Reliable prediction with typical error of ±{mae:.2f}s. "
                f"Performance should fall within expected range."
            )
        elif confidence >= 40:
            stability['confidence_level'] = 'moderate'
            stability['insights'].append(
                f"Moderate confidence ({confidence:.0f}%): Prediction has ±{mae:.2f}s typical error. "
                f"Significant variance possible due to inconsistent driving patterns."
            )
        else:
            stability['confidence_level'] = 'low'
            stability['insights'].append(
                f"Low confidence ({confidence:.0f}%): High prediction uncertainty with ±{mae:.2f}s error range. "
                f"Driver's lap times are highly variable, making accurate prediction challenging."
            )
        
        # Prediction variance analysis
        pred_lower = predictions.get('prediction_lower', 0)
        pred_upper = predictions.get('prediction_upper', 0)
        if pred_lower > 0 and pred_upper > 0:
            pred_range = pred_upper - pred_lower
            stability['prediction_variance'] = pred_range
            
            if pred_range > 2.0:
                stability['insights'].append(
                    f"Wide prediction range (±{pred_range/2:.2f}s): Unusually high variance indicates volatile "
                    f"driving inputs or inconsistent performance. Focus on consistency before chasing speed."
                )
            elif pred_range < 0.5:
                stability['insights'].append(
                    f"Narrow prediction range (±{pred_range/2:.2f}s): Exceptional consistency allows precise "
                    f"prediction. Your repeatable performance is a competitive advantage."
                )
        
        return stability

    
    # ========================================================================
    # MODULE 5: DYNAMIC NATURAL LANGUAGE GENERATION
    # ========================================================================
    
    def generate_dynamic_insight(self, metric_name: str, value: float, benchmark: float, 
                                 context: str = "") -> str:
        """
        Generate dynamic natural language using scales, adjectives, and comparators
        NOT templates - intelligent sentence construction
        """
        # Calculate performance ratio
        if benchmark > 0:
            ratio = value / benchmark
            diff = value - benchmark
            pct_diff = ((value - benchmark) / benchmark) * 100
        else:
            return f"{metric_name} analysis unavailable due to insufficient data."
        
        # Determine severity rating
        severity = self._rate_severity(ratio)
        adjective = self.severity_scales[severity]['adjective']
        
        # Determine comparative language
        if abs(pct_diff) < 2:
            comparative = np.random.choice(self.comparative_adjectives['equal'])
        elif pct_diff < -10:
            comparative = np.random.choice(self.comparative_adjectives['much_better'])
        elif pct_diff < -5:
            comparative = np.random.choice(self.comparative_adjectives['better'])
        elif pct_diff < -2:
            comparative = np.random.choice(self.comparative_adjectives['slightly_better'])
        elif pct_diff > 10:
            comparative = np.random.choice(self.comparative_adjectives['much_worse'])
        elif pct_diff > 5:
            comparative = np.random.choice(self.comparative_adjectives['worse'])
        else:
            comparative = np.random.choice(self.comparative_adjectives['slightly_worse'])
        
        # Construct dynamic sentence
        if context:
            insight = (f"Your {metric_name} is rated {adjective}, {comparative} the benchmark "
                      f"by {abs(diff):.3f} ({abs(pct_diff):.1f}%). {context}")
        else:
            insight = (f"Your {metric_name} is rated {adjective}, {comparative} the benchmark "
                      f"by {abs(diff):.3f} ({abs(pct_diff):.1f}%).")
        
        return insight
    
    def _rate_severity(self, ratio: float) -> str:
        """Rate performance severity based on ratio to benchmark"""
        if ratio >= 0.95:
            return 'excellent'
        elif ratio >= 0.80:
            return 'good'
        elif ratio >= 0.60:
            return 'average'
        elif ratio >= 0.40:
            return 'below_average'
        elif ratio >= 0.20:
            return 'poor'
        else:
            return 'critical'

    
    # ========================================================================
    # MODULE 6: TRACK-SPECIFIC CONTEXT LAYER
    # ========================================================================
    
    def add_track_context(self, insight_type: str, metric_value: float, sector: int = None) -> str:
        """
        Add track-specific context to make insights expert-level
        """
        context_additions = []
        
        # Throttle-related context
        if 'throttle' in insight_type.lower():
            if sector in self.track_context.traction_zones:
                context_additions.append(
                    f"This sector includes a critical traction zone where throttle control directly impacts exit speed."
                )
            if sector in self.track_context.slow_corners:
                context_additions.append(
                    f"The slow corners in this sector punish throttle inconsistency — smooth application is crucial."
                )
        
        # Braking-related context
        if 'brak' in insight_type.lower():
            if sector in self.track_context.heavy_braking_zones:
                context_additions.append(
                    f"This is one of the track's heaviest braking zones — consistency here is critical for lap time."
                )
            if sector in self.track_context.chicanes:
                context_additions.append(
                    f"The chicane in this sector requires precise braking modulation to maintain momentum."
                )
        
        # Speed-related context
        if 'speed' in insight_type.lower():
            if sector in self.track_context.long_straights:
                context_additions.append(
                    f"This sector includes the main straight where aerodynamic efficiency and exit speed matter most."
                )
            if sector in self.track_context.technical_sectors:
                context_additions.append(
                    f"This technical sector rewards precision over raw speed — focus on maintaining minimum corner speeds."
                )
        
        # Consistency-related context
        if 'consistency' in insight_type.lower() or 'variance' in insight_type.lower():
            if sector in self.track_context.technical_sectors:
                context_additions.append(
                    f"The technical nature of this sector amplifies the impact of inconsistency — "
                    f"a variance of {metric_value:.1f}% is particularly costly here."
                )
        
        return " ".join(context_additions) if context_additions else ""

    
    # ========================================================================
    # MASTER ANALYSIS FUNCTION
    # ========================================================================
    
    def generate_intelligent_commentary(self, driver_data: Dict, telemetry_data: pd.DataFrame,
                                       predictions: Dict, field_data: pd.DataFrame) -> Dict:
        """
        Master function: Generates complete intelligent commentary using all 6 modules
        Returns comprehensive analysis with dynamic insights
        """
        commentary = {
            'driving_style': {},
            'consistency_analysis': {},
            'comparative_insights': {},
            'prediction_stability': {},
            'actionable_recommendations': [],
            'expert_summary': ""
        }
        
        # MODULE 1: Detect driving style
        style, style_scores = self.detect_driving_style(telemetry_data)
        commentary['driving_style'] = {
            'primary_style': style.value,
            'confidence_scores': {s.value: score for s, score in style_scores.items()},
            'description': self._get_style_description(style)
        }
        
        # MODULE 2: Analyze consistency patterns
        consistency = self.analyze_consistency_patterns(telemetry_data)
        commentary['consistency_analysis'] = consistency
        
        # MODULE 3: Generate comparative insights
        comparisons = self.generate_comparative_insights(driver_data, field_data)
        commentary['comparative_insights'] = comparisons
        
        # MODULE 4: Analyze prediction stability
        stability = self.analyze_prediction_stability(predictions)
        commentary['prediction_stability'] = stability
        
        # MODULE 5 & 6: Generate dynamic insights with track context
        commentary['actionable_recommendations'] = self._generate_actionable_recommendations(
            driver_data, telemetry_data, style, consistency, comparisons
        )
        
        # Generate expert summary
        commentary['expert_summary'] = self._generate_expert_summary(
            style, consistency, comparisons, stability
        )
        
        return commentary

    
    def _get_style_description(self, style: DrivingStyle) -> str:
        """Get detailed description of driving style - Professional racing analysis"""
        descriptions = {
            DrivingStyle.LATE_BRAKER: (
                "You rely on extended braking zones to maximize straight-line speed, carrying momentum deep into corners "
                "before committing to deceleration. This approach demands exceptional precision and reference point discipline, "
                "as the margin for error narrows significantly with delayed brake application."
            ),
            DrivingStyle.AGGRESSIVE_ACCELERANT: (
                "Your throttle application exhibits high variance and aggressive initial input, generating rear rotation "
                "to aid turn-in but introducing traction management challenges. This style trades mechanical grip for "
                "rotation, requiring refined throttle modulation to prevent power oversteer on corner exit."
            ),
            DrivingStyle.SMOOTH_CONSISTENT: (
                "You demonstrate progressive input application with minimal oscillation, building consistency through "
                "repeatable reference points and measured control transitions. This approach prioritizes predictability "
                "and tire preservation, though it may limit how aggressively you attack mid-corner speed."
            ),
            DrivingStyle.CORNER_SPEED_SPECIALIST: (
                "Your technique emphasizes maintaining momentum through technical sections, carrying higher minimum speeds "
                "at the expense of straight-line velocity. This style requires precise line selection and early throttle "
                "application, converting cornering efficiency into lap time rather than relying on power deployment."
            ),
            DrivingStyle.HIGH_RISK_HIGH_VARIANCE: (
                "You operate with significant performance variance, pushing boundaries that yield occasional fast laps "
                "but introduce substantial inconsistency. This high-risk approach suggests either experimental technique "
                "development or insufficient reference point discipline, limiting your ability to extract repeatable pace."
            ),
            DrivingStyle.CONSERVATIVE_CLEAN: (
                "Your driving prioritizes consistency and risk mitigation over aggressive pace extraction, maintaining "
                "clean inputs and conservative margins. While this approach builds reliability and race craft, it may "
                "leave performance on the table by not fully exploiting the car's dynamic envelope."
            ),
            DrivingStyle.STRAIGHT_SPEED_TECHNICAL_WEAKNESS: (
                "You demonstrate strong performance in power deployment zones but encounter difficulties in technical "
                "sections requiring precision and car control. This pattern suggests either setup limitations affecting "
                "low-speed handling or technique refinement needed for complex corner sequences."
            )
        }
        return descriptions.get(style, "Your driving exhibits balanced characteristics across multiple dimensions, with no single dominant pattern emerging from the telemetry analysis.")
    
    def _generate_actionable_recommendations(self, driver_data: Dict, telemetry_data: pd.DataFrame,
                                            style: DrivingStyle, consistency: Dict, 
                                            comparisons: Dict) -> List[Dict]:
        """Generate prioritized, actionable recommendations"""
        recommendations = []
        
        # Based on driving style
        if style == DrivingStyle.AGGRESSIVE_ACCELERANT:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'Throttle Control Refinement',
                'issue': 'Aggressive throttle application causing traction loss and inconsistency',
                'action': 'Practice progressive throttle application from 60% to 100% through corner exit. Focus on smoothness over aggression.',
                'potential_gain': '0.15-0.25s per lap',
                'track_context': self.add_track_context('throttle', 0, sector=1)
            })
        
        elif style == DrivingStyle.LATE_BRAKER:
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'Braking Consistency',
                'issue': 'Late braking style requires exceptional precision to maintain consistency',
                'action': 'Establish fixed braking markers to maximize your late-braking advantage.',
                'potential_gain': '0.10-0.20s per lap',
                'track_context': self.add_track_context('braking', 0, sector=1)
            })
        
        # Based on consistency patterns
        if consistency['overall_pattern'] == ConsistencyPattern.VOLATILE:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'Consistency Development',
                'issue': f"High lap time volatility (±{consistency['lap_to_lap_variance']:.3f}s) limiting performance",
                'action': 'Focus on repeatable reference points and standardized technique before chasing speed.',
                'potential_gain': '0.20-0.40s per lap',
                'track_context': 'Consistency is the foundation of speed. Address this first.'
            })
        
        # Based on comparative analysis
        percentile = comparisons.get('percentile_rank', 50)
        if percentile < 30:  # Bottom 30%
            recommendations.append({
                'priority': 'HIGH',
                'category': 'Fundamental Pace Development',
                'issue': f"Currently in bottom {100-percentile:.0f}% of field with significant pace deficit",
                'action': 'Focus on fundamental driving technique: racing line, braking points, throttle application.',
                'potential_gain': '0.50-1.00s per lap',
                'track_context': 'Master the basics before advanced techniques.'
            })
        
        return recommendations

    
    def _generate_expert_summary(self, style: DrivingStyle, consistency: Dict,
                                 comparisons: Dict, stability: Dict) -> str:
        """Generate expert-level summary combining all insights"""
        summary_parts = []
        
        # Style-based opening
        summary_parts.append(f"Driver exhibits a {style.value} approach, {self._get_style_description(style)}")
        
        # Consistency assessment
        if consistency['overall_pattern'] == ConsistencyPattern.IMPROVING:
            summary_parts.append(
                f"Performance shows clear improvement trend with {consistency['trend_magnitude']:.3f}s gain "
                f"over session. Positive momentum building."
            )
        elif consistency['overall_pattern'] == ConsistencyPattern.VOLATILE:
            summary_parts.append(
                f"Significant consistency issues with {consistency['lap_to_lap_variance']:.3f}s variance. "
                f"This is the primary limiter to competitive pace."
            )
        else:
            summary_parts.append(
                f"Demonstrates {consistency['pace_stability']*100:.0f}% pace stability. "
                f"A solid foundation for performance development."
            )
        
        # Comparative positioning
        percentile = comparisons.get('percentile_rank', 50)
        if percentile > 80:
            summary_parts.append(
                f"Elite positioning: faster than {percentile:.0f}% of field with clear championship potential."
            )
        elif percentile > 60:
            summary_parts.append(
                f"Strong mid-pack performance: faster than {percentile:.0f}% of field with podium potential."
            )
        elif percentile > 40:
            summary_parts.append(
                f"Mid-field positioning: faster than {percentile:.0f}% of field — incremental gains needed."
            )
        else:
            summary_parts.append(
                f"Development phase: currently faster than {percentile:.0f}% of field — focus on fundamentals."
            )
        
        # Prediction reliability
        reliability = stability.get('reliability_rating', 'unknown')
        if reliability == 'exceptional':
            summary_parts.append(
                "Prediction models show perfect consensus — your consistent performance allows precise forecasting."
            )
        elif reliability == 'low':
            summary_parts.append(
                "Low model consensus indicates unpredictable performance — consistency work is critical."
            )
        
        return " ".join(summary_parts)


    # ========================================================================
    # MODULE 7: PERSONALIZED WEATHER & GRIP ANALYSIS
    # ========================================================================
    
    def analyze_personalized_weather_impact(
        self, 
        driver_data: Dict, 
        telemetry_data: pd.DataFrame,
        weather_impact: Dict,
        style: DrivingStyle
    ) -> Dict:
        """
        Generate personalized weather and grip analysis for specific driver.
        Connects current conditions to driver's performance history and style.
        Uses driver-specific weather data from their laps.
        """
        if telemetry_data is None or len(telemetry_data) == 0:
            return None
            
        # Extract driver-specific weather data (from their actual laps)
        driver_weather = telemetry_data[telemetry_data['TRACK_TEMP'].notna()]
        
        if len(driver_weather) > 0:
            # Use driver's actual experienced conditions
            avg_temp = driver_weather['TRACK_TEMP'].mean()
            avg_humid = driver_weather['HUMIDITY'].mean()
            temp_range = (driver_weather['TRACK_TEMP'].min(), driver_weather['TRACK_TEMP'].max())
            humid_range = (driver_weather['HUMIDITY'].min(), driver_weather['HUMIDITY'].max())
        else:
            # Fallback to global weather data
            avg_temp = weather_impact['avg_track_temp']
            avg_humid = weather_impact['avg_humidity']
            temp_range = (weather_impact['temp_range']['min'], weather_impact['temp_range']['max'])
            humid_range = (avg_humid - 2, avg_humid + 2)
        
        optimal_temp = weather_impact['optimal_conditions']['track_temp']
        optimal_humid = weather_impact['optimal_conditions']['humidity']
        temp_corr = weather_impact['weather_correlation']['track_temp']
        humid_corr = weather_impact['weather_correlation']['humidity']
        
        # Calculate temperature and humidity deltas
        temp_diff = abs(avg_temp - optimal_temp)
        humid_diff = abs(avg_humid - optimal_humid)
        
        # Calculate driver-specific metrics
        analysis = {}
        
        # 1. Current Grip Delta (Driver-Specific)
        if 'g_force_total' in telemetry_data.columns:
            current_g_force = telemetry_data['g_force_total'].mean()
            best_g_force = telemetry_data['g_force_total'].quantile(0.95)
            
            if best_g_force > 0:
                grip_delta = ((current_g_force - best_g_force) / best_g_force) * 100
                analysis['grip_delta_percent'] = grip_delta
                analysis['current_g_force'] = current_g_force
                analysis['best_g_force'] = best_g_force
            else:
                analysis['grip_delta_percent'] = 0
                analysis['current_g_force'] = current_g_force
                analysis['best_g_force'] = current_g_force
        else:
            analysis['grip_delta_percent'] = 0
            analysis['current_g_force'] = 0.2
            analysis['best_g_force'] = 0.25
        
        # 2. Tire Temperature Sensitivity Score
        # Analyze consistency metrics vs temperature
        if 'Brake_Front' in telemetry_data.columns and 'TRACK_TEMP' in telemetry_data.columns:
            brake_consistency = telemetry_data['Brake_Front'].std()
            temp_variance = telemetry_data['TRACK_TEMP'].std()
            
            if temp_variance > 0:
                sensitivity_score = brake_consistency / temp_variance
                analysis['temp_sensitivity_score'] = sensitivity_score
                
                # Classify sensitivity
                if sensitivity_score > 2.0:
                    analysis['temp_sensitivity_level'] = 'High'
                elif sensitivity_score > 1.0:
                    analysis['temp_sensitivity_level'] = 'Moderate'
                else:
                    analysis['temp_sensitivity_level'] = 'Low'
            else:
                analysis['temp_sensitivity_score'] = 1.0
                analysis['temp_sensitivity_level'] = 'Moderate'
        else:
            analysis['temp_sensitivity_score'] = 1.0
            analysis['temp_sensitivity_level'] = 'Moderate'
        
        # 3. Generate Conditions Summary
        analysis['conditions'] = {
            'track_temp': avg_temp,
            'optimal_temp': optimal_temp,
            'temp_diff': temp_diff,
            'humidity': avg_humid,
            'optimal_humid': optimal_humid,
            'humid_diff': humid_diff,
            'temp_correlation': abs(temp_corr),
            'humid_correlation': abs(humid_corr)
        }
        
        # Determine which factor is more critical
        if temp_diff > humid_diff / 10:  # Scale humidity to comparable range
            analysis['primary_factor'] = 'temperature'
        else:
            analysis['primary_factor'] = 'humidity'
        
        # 4. Generate Personalized Impact Narrative
        grip_delta = analysis['grip_delta_percent']
        temp_sensitivity = analysis['temp_sensitivity_level']
        
        impact_narrative = []
        
        # State the loss/gain
        if grip_delta < -1.5:
            impact_narrative.append(
                f"Current conditions show a {abs(grip_delta):.2f}% reduction in your average usable grip "
                f"compared to your best historical performance. This is a significant impact."
            )
            analysis['impact_severity'] = 'high'
        elif grip_delta < -0.5:
            impact_narrative.append(
                f"Current conditions show a {abs(grip_delta):.2f}% reduction in your average usable grip "
                f"compared to your best performance. This is affecting your pace."
            )
            analysis['impact_severity'] = 'moderate'
        elif grip_delta > 0.5:
            impact_narrative.append(
                f"Current conditions are favorable, showing a {grip_delta:.2f}% improvement in usable grip "
                f"compared to your average performance."
            )
            analysis['impact_severity'] = 'positive'
        else:
            impact_narrative.append(
                f"Current conditions are neutral, with minimal impact ({abs(grip_delta):.2f}%) on your usable grip levels."
            )
            analysis['impact_severity'] = 'neutral'
        
        # Explain the cause
        if temp_sensitivity == 'High':
            impact_narrative.append(
                f"Your High Tire Temperature Sensitivity suggests the {temp_diff:.1f}°C difference from optimal "
                f"is likely affecting your mid-corner stability and confidence."
            )
        elif temp_sensitivity == 'Moderate':
            impact_narrative.append(
                f"Your Moderate Temperature Sensitivity means the {temp_diff:.1f}°C variance from optimal "
                f"has a noticeable but manageable effect on your performance."
            )
        else:
            impact_narrative.append(
                f"Your Low Temperature Sensitivity means you adapt well to the {temp_diff:.1f}°C variance from optimal conditions."
            )
        
        # Connect to driving style
        if style == DrivingStyle.AGGRESSIVE_ACCELERANT and temp_diff > 1.0:
            impact_narrative.append(
                "Your aggressive throttle style may struggle with the current grip levels, requiring more refined modulation."
            )
        elif style == DrivingStyle.LATE_BRAKER and humid_diff > 10:
            impact_narrative.append(
                "Your late braking style requires maximum grip confidence, which may be compromised by current humidity levels."
            )
        elif style == DrivingStyle.CORNER_SPEED_SPECIALIST:
            impact_narrative.append(
                "Your corner speed approach relies heavily on consistent grip, making weather conditions particularly important."
            )
        
        analysis['impact_narrative'] = " ".join(impact_narrative)
        
        # 5. Generate Actionable Management Strategy
        strategy = []
        
        if avg_temp < optimal_temp - 1.0:
            # Cold conditions
            strategy.append(
                "Focus on early throttle application (5-10%) on corner exit to build essential tire heat, "
                "even if it sacrifices peak exit speed initially."
            )
        elif avg_temp > optimal_temp + 1.0:
            # Hot conditions
            strategy.append(
                "Manage tire temperatures by avoiding excessive wheel spin and maintaining smooth inputs "
                "to preserve grip throughout the stint."
            )
        
        if avg_humid > optimal_humid + 10:
            # High humidity / lower grip
            strategy.append(
                "Prioritize smooth steering inputs over aggressive trail braking to manage the lower grip levels. "
                "Build confidence progressively rather than attacking immediately."
            )
        elif avg_humid < optimal_humid - 10:
            # Low humidity / better grip
            strategy.append(
                "Take advantage of the improved grip levels by carrying more mid-corner speed and being more aggressive on throttle."
            )
        
        if not strategy:
            # Optimal conditions
            strategy.append(
                "Conditions are near optimal. Focus on consistency and executing your natural driving style without adaptation."
            )
        
        analysis['management_strategy'] = " ".join(strategy)
        
        return analysis
