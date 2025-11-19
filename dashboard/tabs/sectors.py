"""Sector Analysis tab - Comprehensive Sector Performance Analysis with Enhanced Timing."""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import numpy as np
from typing import Dict, Any

def _color_for_gap(gap: float) -> str:
    """Return hex color for gap value (positive gap means slower)."""
    if gap >= 0.30:
        return "rgba(220, 53, 69, 0.8)"  # Danger red
    if gap >= 0.10:
        return "rgba(255, 193, 7, 0.8)"  # Warning amber
    return "rgba(0, 191, 255, 0.6)"      # Cyan for small gap

def _format_seconds(v: float) -> str:
    return f"{v:.3f}s"

def render(selected_driver, selected_driver_name, enhanced_mode, df, COLORS, apply_standard_layout, RESPONSIVE_CONFIG, predictions, commentary, clusters, model_selection, driver_commentary, driver_clusters):
    """Render the comprehensive Sector Analysis tab content with enhanced timing features."""

    # Get project root for absolute paths
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Load basic sector data
    try:
        sector_stats = pd.read_csv(os.path.join(project_root, 'outputs/sector_analysis/sector_statistics.csv'))
        sector_data = pd.read_csv(os.path.join(project_root, 'outputs/sector_analysis/sector_data_detailed.csv'))
        has_sector_data = True
    except Exception as e:
        has_sector_data = False
        st.warning(f"⚠️ Sector analysis data not available. Run: py scripts/analyze_sectors.py")

    # Load enhanced sector data (6 timing points)
    try:
        timing_recs = pd.read_csv(os.path.join(project_root, 'outputs/enhanced_sector_analysis/timing_recommendations.csv'))
    except Exception:
        timing_recs = pd.DataFrame()

    try:
        with open(os.path.join(project_root, 'outputs/enhanced_sector_analysis/full_analysis.json'), 'r') as f:
            full_analysis = json.load(f)
        has_enhanced_data = True
    except Exception as e:
        full_analysis = []
        has_enhanced_data = False
        pass  # Enhanced data not available

    # Helper: extract selected driver number from format "GR86-XXX-YY"
    def _extract_driver_number(name: str):
        try:
            return int(name.split('-')[-1])
        except Exception:
            try:
                return int(''.join(filter(str.isdigit, name)))
            except Exception:
                return None

    if has_sector_data:
        driver_num = _extract_driver_number(selected_driver_name)
        driver_sector = sector_stats[sector_stats['driver_number'] == driver_num]

        if len(driver_sector) > 0:
            driver_sector = driver_sector.iloc[0]

            # Start with high-level summary (Best & Worst Sectors)
            st.markdown("### Best & Worst Sectors")
            col1, col2 = st.columns(2)

            # compute strongest gap display value
            strongest_sector = driver_sector.get('strongest_sector', 'S1')
            weakest_sector = driver_sector.get('weakest_sector', 'S3')

            strongest_gap = (
                driver_sector.get('s1_gap_to_best', np.nan)
                if strongest_sector == 'S1'
                else driver_sector.get('s2_gap_to_best', np.nan)
                if strongest_sector == 'S2'
                else driver_sector.get('s3_gap_to_best', np.nan)
            )

            with col1:
                st.markdown(f"""
                <div style="background: rgba(44, 62, 80, 0.95); 
                     padding: 1rem 1.25rem; 
                     margin: 0.5rem 0; 
                     border-radius: 12px; 
                     border-left: 4px solid rgba(0, 191, 255, 0.6);">
                    <div style="font-size:0.85rem;color:#B0B0B0;font-weight:600;text-transform:uppercase;">Strongest Sector</div>
                    <div style="font-size:1.6rem;font-weight:700;color:#FFFFFF;margin-top:0.25rem;">{strongest_sector}</div>
                    <div style="color:#E8E8E8;margin-top:0.5rem;">Performs closest to overall best in this sector<br/>Gap to best: <strong style='color:#FFFFFF'>{strongest_gap:.3f}s</strong></div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div style="background: rgba(44, 62, 80, 0.95); 
                     padding: 1rem 1.25rem; 
                     margin: 0.5rem 0; 
                     border-radius: 12px; 
                     border-left: 4px solid rgba(0, 191, 255, 0.6);">
                    <div style="font-size:0.85rem;color:#B0B0B0;font-weight:600;text-transform:uppercase;">Improvement Area</div>
                    <div style="font-size:1.6rem;font-weight:700;color:#FFFFFF;margin-top:0.25rem;">{weakest_sector}</div>
                    <div style="color:#E8E8E8;margin-top:0.5rem;">Focus on this sector for maximum gains<br/>Potential: <strong style='color:#FFFFFF'>{driver_sector.get('improvement_potential', 0):.3f}s per lap</strong></div>
                </div>
                """, unsafe_allow_html=True)

            # Sector Times (S1 / S2 / S3)
            st.markdown('<hr class="divider-subtle">', unsafe_allow_html=True)
            st.markdown("### Sector Times")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Sector 1",
                    f"{driver_sector['s1_best']:.3f}s",
                    delta=f"Avg: {driver_sector['s1_avg']:.3f}s",
                    help=f"Best S1 time | Consistency: {driver_sector['s1_consistency']:.1f}%"
                )
            with col2:
                st.metric(
                    "Sector 2",
                    f"{driver_sector['s2_best']:.3f}s",
                    delta=f"Avg: {driver_sector['s2_avg']:.3f}s",
                    help=f"Best S2 time | Consistency: {driver_sector['s2_consistency']:.1f}%"
                )
            with col3:
                st.metric(
                    "Sector 3",
                    f"{driver_sector['s3_best']:.3f}s",
                    delta=f"Avg: {driver_sector['s3_avg']:.3f}s",
                    help=f"Best S3 time | Consistency: {driver_sector['s3_consistency']:.1f}%"
                )

            # Sector comparison chart
            st.markdown('<hr class="divider-glow">', unsafe_allow_html=True)
            st.markdown("### Sector Comparison")
            st.markdown("""
            <div style="background: rgba(44, 62, 80, 0.5); padding: 1rem; border-radius: 8px; margin-bottom: 1rem; border-left: 4px solid #00BFFF;">
                <strong style="color: #00BFFF;">How to Read This Chart:</strong><br/>
                <span style="color: #E8E8E8; font-size: 0.9rem;">
                Each driver shows 3 colored segments stacking up to their total lap time. Shorter bars = faster times. Your selected driver is highlighted.
                </span>
            </div>
            """, unsafe_allow_html=True)

            selected_driver_num = driver_num
            selected_driver_row = sector_stats[sector_stats['driver_number'] == selected_driver_num]

            if len(selected_driver_row) > 0:
                other_drivers = sector_stats[sector_stats['driver_number'] != selected_driver_num].head(7)
                display_drivers = pd.concat([selected_driver_row, other_drivers]).sort_values('lap_best').reset_index(drop=True)
            else:
                display_drivers = sector_stats.head(8)

            fig = go.Figure()
            for idx, row in display_drivers.iterrows():
                driver_num_str = str(int(row['driver_number']))
                is_selected = row['driver_number'] == selected_driver_num
                border_width = 3 if is_selected else 1
                border_color = '#FFFFFF' if is_selected else 'rgba(255, 255, 255, 0.2)'

                fig.add_trace(go.Bar(
                    name='Sector 1' if idx == 0 else '',
                    x=[driver_num_str],
                    y=[row['s1_best']],
                    marker=dict(color='rgba(255, 193, 7, 0.8)', line=dict(color=border_color, width=border_width)),
                    text=f"S1: {row['s1_best']:.2f}s",
                    textposition='inside',
                    textfont=dict(size=10, color='#FFFFFF'),
                    hovertemplate=f'<b>Driver {driver_num_str}</b><br>Sector 1: {row["s1_best"]:.2f}s<extra></extra>',
                    showlegend=(idx == 0),
                    legendgroup='S1'
                ))
                fig.add_trace(go.Bar(
                    name='Sector 2' if idx == 0 else '',
                    x=[driver_num_str],
                    y=[row['s2_best']],
                    marker=dict(color='rgba(0, 191, 255, 0.8)', line=dict(color=border_color, width=border_width)),
                    text=f"S2: {row['s2_best']:.2f}s",
                    textposition='inside',
                    textfont=dict(size=10, color='#FFFFFF'),
                    hovertemplate=f'<b>Driver {driver_num_str}</b><br>Sector 2: {row["s2_best"]:.2f}s<extra></extra>',
                    showlegend=(idx == 0),
                    legendgroup='S2'
                ))
                fig.add_trace(go.Bar(
                    name='Sector 3' if idx == 0 else '',
                    x=[driver_num_str],
                    y=[row['s3_best']],
                    marker=dict(color='rgba(40, 167, 69, 0.8)', line=dict(color=border_color, width=border_width)),
                    text=f"S3: {row['s3_best']:.2f}s",
                    textposition='inside',
                    textfont=dict(size=10, color='#FFFFFF'),
                    hovertemplate=f'<b>Driver {driver_num_str}</b><br>Sector 3: {row["s3_best"]:.2f}s<extra></extra>',
                    showlegend=(idx == 0),
                    legendgroup='S3'
                ))

            fig.update_layout(
                barmode='stack',
                bargap=0.15,
                xaxis_title=dict(text="<b>Driver Number (Ranked by Best Lap)</b>", font=dict(size=11, color='#FFFFFF')),
                yaxis_title=dict(text="<b>Cumulative Time (seconds)</b>", font=dict(size=11, color='#FFFFFF')),
                height=550,
                plot_bgcolor='rgba(26, 37, 47, 0.5)',
                paper_bgcolor='rgba(44, 62, 80, 0.3)',
                font=dict(color='#FFFFFF'),
                margin=dict(t=120, b=100, l=60, r=20),
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.25,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=12, color='#FFFFFF'),
                    bgcolor='rgba(44, 62, 80, 0.9)',
                    bordercolor='rgba(0, 191, 255, 0.5)',
                    borderwidth=1
                ),
                xaxis=dict(showgrid=False, tickfont=dict(size=11, color='#FFFFFF'), showline=True, linewidth=2, linecolor='rgba(0, 191, 255, 0.5)', type='category'),
                yaxis=dict(showgrid=True, gridcolor='rgba(255, 255, 255, 0.1)', tickfont=dict(size=11, color='#FFFFFF'), showline=True, linewidth=2, linecolor='rgba(0, 191, 255, 0.5)')
            )

            chart_config = {
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToRemove': ['lasso2d', 'select2d', 'autoScale2d'],
                'responsive': True,
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': f'sector_comparison_{selected_driver_name}',
                    'height': 550,
                    'width': 1200,
                    'scale': 2
                }
            }

            st.plotly_chart(fig, use_container_width=True, config=chart_config, key="sector_comparison")

            # Radar chart
            st.markdown('<hr class="divider-glow">', unsafe_allow_html=True)
            st.markdown("### Sector Performance Breakdown")
            st.markdown("""
            <div style="background: rgba(44, 62, 80, 0.5); padding: 1rem; border-radius: 8px; margin-bottom: 1rem; border-left: 4px solid #00BFFF;">
                <strong style="color: #00BFFF;">How to Read This Chart:</strong><br/>
                <span style="color: #E8E8E8; font-size: 0.9rem;">
                A perfect triangle = 100% consistency across all sectors. Larger shapes = slower times. Compare your shape with top drivers to see strengths and weaknesses.
                </span>
            </div>
            """, unsafe_allow_html=True)

            top_3 = sector_stats.head(3)
            fig = go.Figure()

            # Add selected driver
            fig.add_trace(go.Scatterpolar(
                r=[driver_sector['s1_best'], driver_sector['s2_best'], driver_sector['s3_best']],
                theta=['Sector 1', 'Sector 2', 'Sector 3'],
                fill='toself',
                name=f'Driver #{selected_driver_num} (You)',
                line=dict(color='#00BFFF', width=4),
                fillcolor='rgba(0, 191, 255, 0.2)',
                hovertemplate='<b>You</b><br>%{theta}: <b>%{r:.3f}s</b><extra></extra>',
                opacity=1.0
            ))

            colors_radar = ['#FFD700', '#C0C0C0', '#CD7F32']
            labels_radar = ['1st Place', '2nd Place', '3rd Place']
            for idx, (_, driver) in enumerate(top_3.iterrows()):
                fig.add_trace(go.Scatterpolar(
                    r=[driver['s1_best'], driver['s2_best'], driver['s3_best']],
                    theta=['Sector 1', 'Sector 2', 'Sector 3'],
                    fill='toself',
                    name=f"{labels_radar[idx]} - Driver #{int(driver['driver_number'])}",
                    line=dict(color=colors_radar[idx], width=3),
                    fillcolor='rgba(255, 215, 0, 0.08)' if idx == 0 else 'rgba(192,192,192,0.06)',
                    hovertemplate=f'<b>{labels_radar[idx]}</b><br>%{{theta}}: <b>%{{r:.3f}}s</b><extra></extra>',
                    opacity=0.85
                ))

            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, max(sector_stats.head(15)[['s1_best', 's2_best', 's3_best']].max()) * 1.1],
                        gridcolor='rgba(255, 255, 255, 0.2)',
                        tickfont=dict(color='#FFFFFF', size=10)
                    ),
                    angularaxis=dict(
                        gridcolor='rgba(255, 255, 255, 0.2)',
                        tickfont=dict(color='#FFFFFF', size=12)
                    ),
                    bgcolor='rgba(26, 37, 47, 0.5)'
                ),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5, bgcolor='rgba(44, 62, 80, 0.9)', bordercolor='rgba(0, 191, 255, 0.5)', borderwidth=2, font=dict(size=11, color='#FFFFFF')),
                paper_bgcolor='rgba(44, 62, 80, 0.3)',
                plot_bgcolor='rgba(26, 37, 47, 0.5)',
                font=dict(color='#FFFFFF'),
                height=500,
                margin=dict(t=120, b=100, l=80, r=80)
            )

            st.plotly_chart(fig, width="stretch", config=chart_config, key="sector_radar")

            # Consistency Score chart (kept but ensure ascending/reversed values logic is correct)
            st.markdown('<hr class="divider-glow">', unsafe_allow_html=True)
            st.markdown("### Consistency Score")
            st.markdown("""
            <div style="background: rgba(44, 62, 80, 0.5); padding: 1rem; border-radius: 8px; margin-bottom: 1rem; border-left: 4px solid rgba(0, 191, 255, 0.6);">
                <strong style="color: #00BFFF;">Consistency Score:</strong><br/>
                <span style="color: #E8E8E8; font-size: 0.9rem;">
                Measures how close a driver's lap times are to each other. Higher % = more consistent performance.
                </span>
            </div>
            """, unsafe_allow_html=True)

            fig = go.Figure()
            # Use largest consistency as "better" - if dataset uses "lower is better" invert accordingly
            # Here assume 'overall_consistency' is percentage (higher = better)
            consistency_sorted = sector_stats.sort_values('overall_consistency', ascending=True).head(10)
            fig.add_trace(go.Bar(
                y=consistency_sorted['driver_number'].astype(str),
                x=consistency_sorted['overall_consistency'],
                orientation='h',
                marker=dict(color=consistency_sorted['overall_consistency'], colorscale='RdYlGn', showscale=True, colorbar=dict(title=dict(text="Consistency<br>Score (%)", font=dict(color='#FFFFFF', size=11)), tickfont=dict(color='#FFFFFF', size=10)), line=dict(color='rgba(0, 0, 0, 0.3)', width=1)),
                text=[f"<b>{v:.1f}%</b>" for v in consistency_sorted['overall_consistency']],
                textposition='outside',
                textfont=dict(size=10, color='#FFFFFF'),
                hovertemplate='<b>Driver %{y}</b><br>Consistency: %{x:.1f}%<extra></extra>'
            ))

            fig.update_layout(
                xaxis_title=dict(text="<b>Consistency Score (%)</b>", font=dict(size=11, color='#FFFFFF')),
                yaxis_title=dict(text="<b>Driver Number</b>", font=dict(size=11, color='#FFFFFF')),
                height=450,
                plot_bgcolor='rgba(26, 37, 47, 0.5)',
                paper_bgcolor='rgba(44, 62, 80, 0.3)',
                font=dict(color='#FFFFFF'),
                margin=dict(t=120, b=60, l=60, r=20),
                yaxis=dict(autorange="reversed", showgrid=False, tickfont=dict(size=11, color='#FFFFFF'), showline=True, linewidth=2, linecolor='rgba(0, 191, 255, 0.5)'),
                xaxis=dict(showgrid=True, gridcolor='rgba(255, 255, 255, 0.1)', tickfont=dict(size=11, color='#FFFFFF'), showline=True, linewidth=2, linecolor='rgba(0, 191, 255, 0.5)')
            )

            st.plotly_chart(fig, width="stretch", config=chart_config, key="sector_consistency")

            # ========================================================================
            # ENHANCED SECTOR ANALYSIS - 6 Timing Points + Recommendations + Wind + Top Speed
            # ========================================================================
            if has_enhanced_data:
                st.markdown('<hr class="divider-glow">', unsafe_allow_html=True)

                # Find driver's enhanced data
                driver_analysis = None
                for driver_data in full_analysis:
                    if driver_data.get('driver_name') == selected_driver_name or str(driver_data.get('driver_number')) in selected_driver_name:
                        driver_analysis = driver_data
                        break

                if driver_analysis is None:
                    st.info("Enhanced timing analysis not available for this driver.")
                else:
                    timing_analysis = driver_analysis.get('timing_analysis', {})
                    timing_points: Dict[str, Any] = timing_analysis.get('timing_points', {})
                    
                    # Add custom CSS for expander styling
                    st.markdown("""
                    <style>
                    .stExpander {
                        background: linear-gradient(135deg, rgba(44, 62, 80, 0.95) 0%, rgba(52, 73, 94, 0.95) 100%);
                        
                        
                        border-radius: 12px;
                        margin: 0.25rem 0;
                    }
                    .stExpander > summary {
                        color: #FFFFFF !important;
                        font-weight: 600;
                    }
                    </style>
                    """, unsafe_allow_html=True)
                    
                    # Detailed Timing Analysis (Collapsible)
                    with st.expander("**Detailed Timing Analysis**", expanded=False):
                        # Map segment keys to display labels
                        expected_points = [
                            ("Start_to_IM1a", "Turn 1 Entry"),
                            ("IM1_to_IM2a", "Turn 7 Complex Entry"),
                            ("IM2_to_IM3a", "Oval Entry"),
                            ("IM3a_to_FL", "Finish Line")
                        ]

                        # Present timing points in 2x2 grid
                        for i in range(0, len(expected_points), 2):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                if i < len(expected_points):
                                    key, label = expected_points[i]
                                    point = timing_points.get(key, {})
                                    avg_time = point.get('avg_time', np.nan)
                                    best_time = point.get('best_time', np.nan)
                                    gap = avg_time - best_time if (not pd.isna(avg_time) and not pd.isna(best_time)) else np.nan

                                    st.markdown(f"""
                                    <div style="background: linear-gradient(135deg, rgba(52, 73, 94, 0.95) 0%, rgba(44, 62, 80, 0.95) 100%); 
                                         padding: 1rem; 
                                         border-radius: 12px; 
                                         border-left: 4px solid rgba(0, 191, 255, 0.6);
                                         border-top: 1px solid rgba(255, 255, 255, 0.1);
                                         border-right: 1px solid rgba(255, 255, 255, 0.1);
                                         border-bottom: 1px solid rgba(255, 255, 255, 0.1);
                                         box-shadow: 0 4px 16px rgba(0, 0, 0, 0.5);
                                         margin: 0.5rem 0;">
                                        <div style="font-size:0.75rem;color:#00BFFF;font-weight:700;text-transform:uppercase;letter-spacing:1px;">{label}</div>
                                        <div style="display:flex;justify-content:space-between;align-items:center;margin-top:0.75rem;">
                                            <div>
                                                <div style="font-size:0.75rem;color:#B0B0B0;font-weight:600;">Your Avg</div>
                                                <div style="font-size:1.4rem;color:#FFFFFF;font-weight:700;margin-top:0.25rem;">{_format_seconds(avg_time) if not pd.isna(avg_time) else '—'}</div>
                                            </div>
                                            <div>
                                                <div style="font-size:0.75rem;color:#B0B0B0;font-weight:600;">Best</div>
                                                <div style="font-size:1.4rem;color:#00BFFF;font-weight:700;margin-top:0.25rem;">{_format_seconds(best_time) if not pd.isna(best_time) else '—'}</div>
                                            </div>
                                            <div style="text-align:right;">
                                                <div style="font-size:0.75rem;color:#B0B0B0;font-weight:600;">Gap</div>
                                                <div style="font-size:1.4rem;font-weight:700;color:#FFFFFF;margin-top:0.25rem;">{('+' if not pd.isna(gap) and gap >= 0 else '') + _format_seconds(gap) if not pd.isna(gap) else '—'}</div>
                                            </div>
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                            
                            with col2:
                                if i + 1 < len(expected_points):
                                    key, label = expected_points[i + 1]
                                    point = timing_points.get(key, {})
                                    avg_time = point.get('avg_time', np.nan)
                                    best_time = point.get('best_time', np.nan)
                                    gap = avg_time - best_time if (not pd.isna(avg_time) and not pd.isna(best_time)) else np.nan

                                    st.markdown(f"""
                                    <div style="background: linear-gradient(135deg, rgba(52, 73, 94, 0.95) 0%, rgba(44, 62, 80, 0.95) 100%); 
                                         padding: 1rem; 
                                         border-radius: 12px; 
                                         border-left: 4px solid rgba(0, 191, 255, 0.6);
                                         border-top: 1px solid rgba(255, 255, 255, 0.1);
                                         border-right: 1px solid rgba(255, 255, 255, 0.1);
                                         border-bottom: 1px solid rgba(255, 255, 255, 0.1);
                                         box-shadow: 0 4px 16px rgba(0, 0, 0, 0.5);
                                         margin: 0.5rem 0;">
                                        <div style="font-size:0.75rem;color:#00BFFF;font-weight:700;text-transform:uppercase;letter-spacing:1px;">{label}</div>
                                        <div style="display:flex;justify-content:space-between;align-items:center;margin-top:0.75rem;">
                                            <div>
                                                <div style="font-size:0.75rem;color:#B0B0B0;font-weight:600;">Your Avg</div>
                                                <div style="font-size:1.4rem;color:#FFFFFF;font-weight:700;margin-top:0.25rem;">{_format_seconds(avg_time) if not pd.isna(avg_time) else '—'}</div>
                                            </div>
                                            <div>
                                                <div style="font-size:0.75rem;color:#B0B0B0;font-weight:600;">Best</div>
                                                <div style="font-size:1.4rem;color:#00BFFF;font-weight:700;margin-top:0.25rem;">{_format_seconds(best_time) if not pd.isna(best_time) else '—'}</div>
                                            </div>
                                            <div style="text-align:right;">
                                                <div style="font-size:0.75rem;color:#B0B0B0;font-weight:600;">Gap</div>
                                                <div style="font-size:1.4rem;font-weight:700;color:#FFFFFF;margin-top:0.25rem;">{('+' if not pd.isna(gap) and gap >= 0 else '') + _format_seconds(gap) if not pd.isna(gap) else '—'}</div>
                                            </div>
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                    
                    # Weakest Segments (Collapsible) - 3 cards in a row
                    weakest_segments = timing_analysis.get('weakest_segments', [])
                    if weakest_segments:
                        with st.expander("**Weakest Segments**", expanded=False):
                            cols = st.columns(3)
                            for idx, seg in enumerate(weakest_segments[:3]):
                                with cols[idx]:
                                    loc = seg.get('location', 'Unknown')
                                    seg_type = seg.get('type', 'unknown').title()
                                    avg_time = seg.get('avg_time', np.nan)
                                    best_time = seg.get('best_time', np.nan)
                                    pot_gain = seg.get('potential_gain', 0)
                                    st.markdown(f"""
                                <div class="weakest-segment-card" style="background: linear-gradient(135deg, rgba(52, 73, 94, 0.95) 0%, rgba(44, 62, 80, 0.95) 100%); 
                                     padding: 1rem 1.25rem; 
                                     border-radius: 12px; 
                                     margin: 0.5rem 0; 
                                     border-left: 4px solid rgba(0, 191, 255, 0.6);
                                     border-top: 1px solid rgba(255, 255, 255, 0.1);
                                     border-right: 1px solid rgba(255, 255, 255, 0.1);
                                     border-bottom: 1px solid rgba(255, 255, 255, 0.1);
                                     box-shadow: 0 4px 16px rgba(0, 0, 0, 0.5);
                                     min-height: 140px;
                                     display: flex;
                                     flex-direction: column;
                                     justify-content: space-between;">
                                    <div style="font-size:0.75rem;color:#00BFFF;font-weight:700;text-transform:uppercase;letter-spacing:1px;">{seg_type} — POTENTIAL GAIN {pot_gain:.4f}s</div>
                                    <div style="font-size:1.1rem;color:#FFFFFF;font-weight:700;margin-top:0.5rem;">{loc}</div>
                                    <div style="font-size:0.85rem;color:#E8E8E8;margin-top:0.5rem;">Avg: <strong style="color:#FFFFFF;font-weight:700;">{_format_seconds(avg_time)}</strong> • Best: <strong style="color:#FFFFFF;font-weight:700;">{_format_seconds(best_time)}</strong></div>
                                </div>
                                """, unsafe_allow_html=True)



                    # Improvement Recommendations (Collapsible) - 3 cards in a row
                    with st.expander("**Improvement Recommendations**", expanded=False):
                        # Match by driver number instead of name
                        driver_recs = timing_recs[timing_recs['driver_number'] == selected_driver_num] if not timing_recs.empty else pd.DataFrame()
                        if len(driver_recs) > 0:
                            cols = st.columns(3)
                            for idx, (_, rec) in enumerate(driver_recs.head(3).iterrows()):
                                with cols[idx]:
                                    priority = rec.get('priority', 'LOW').upper()
                                    pot_gain_str = rec.get('potential_gain', '0s')
                                    loc = rec.get('location', 'Unknown')
                                    issue = rec.get('issue', '')
                                    action = rec.get('action', '')
                                    
                                        # Use consistent cyan color for all priorities
                                    priority_color = "#00BFFF"
                                    
                                    st.markdown(f"""
                                <div class="improvement-rec-card" style="background: linear-gradient(135deg, rgba(52, 73, 94, 0.95) 0%, rgba(44, 62, 80, 0.95) 100%); 
                                     padding: 1rem 1.25rem; 
                                     margin: 0.5rem 0; 
                                     border-radius: 12px; 
                                     border-left: 4px solid rgba(0, 191, 255, 0.6);
                                     border-top: 1px solid rgba(255, 255, 255, 0.1);
                                     border-right: 1px solid rgba(255, 255, 255, 0.1);
                                     border-bottom: 1px solid rgba(255, 255, 255, 0.1);
                                     box-shadow: 0 4px 16px rgba(0, 0, 0, 0.5);
                                     min-height: 200px;
                                     display: flex;
                                     flex-direction: column;
                                     justify-content: space-between;">
                                    <div style="font-size: 0.75rem; 
                                         font-weight: 700; 
                                         color: {priority_color}; 
                                         margin-bottom: 0.5rem; 
                                         letter-spacing: 1px;
                                         text-transform: uppercase;">
                                        {priority} PRIORITY — {pot_gain_str}
                                    </div>
                                    <div style="font-size: 1.1rem; font-weight: 600; color: #FFFFFF; margin-bottom: 0.5rem;">
                                        {loc}
                                    </div>
                                    <div style="font-size: 0.85rem; color: #E8E8E8; margin-bottom: 0.5rem;">
                                        <strong style="color: #00BFFF;">Issue:</strong> {issue}
                                    </div>
                                    <div style="font-size: 0.85rem; color: #E8E8E8;">
                                        <strong style="color: #00BFFF;">Action:</strong> {action}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.info("No specific recommendations available for this driver.")
                    
                    # Aerodynamic & Speed Analysis (before Total Potential Gain)
                    wind_analysis = driver_analysis.get('wind_analysis', {})
                    top_speed_stats = driver_analysis.get('top_speed_stats', {})
                    
                    if wind_analysis or top_speed_stats:
                        with st.expander("**Aerodynamic & Speed Analysis**", expanded=False):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                if wind_analysis:
                                    avg_wind = wind_analysis.get('avg_wind_speed', 0)
                                    severity = wind_analysis.get('severity', 'Low')
                                    impact_est = wind_analysis.get('wind_impact_estimate', 0.0)
                                    st.markdown(f"""
                                    <div style="background: linear-gradient(135deg, rgba(52, 73, 94, 0.95) 0%, rgba(44, 62, 80, 0.95) 100%); 
                                         padding: 1rem 1.25rem; 
                                         margin: 0.5rem 0; 
                                         border-radius: 12px; 
                                         border-left: 4px solid rgba(0, 191, 255, 0.6);
                                         border-top: 1px solid rgba(255, 255, 255, 0.1);
                                         border-right: 1px solid rgba(255, 255, 255, 0.1);
                                         border-bottom: 1px solid rgba(255, 255, 255, 0.1);
                                         box-shadow: 0 4px 16px rgba(0, 0, 0, 0.5);">
                                        <div style="font-size: 0.75rem; 
                                             font-weight: 700; 
                                             color: #00BFFF; 
                                             margin-bottom: 0.5rem; 
                                             letter-spacing: 1px;
                                             text-transform: uppercase;">
                                            Wind Impact
                                        </div>
                                        <div style="font-size: 1.4rem; font-weight: 700; color: #FFFFFF; margin-bottom: 0.5rem;">
                                            {avg_wind:.1f} km/h
                                        </div>
                                        <div style="font-size: 0.85rem; color: #E8E8E8; line-height: 1.6;">
                                            Severity: <strong style="color: #FFFFFF; font-weight: 700;">{severity}</strong><br/>
                                            Estimated Impact: <strong style="color: #FFFFFF; font-weight: 700;">±{impact_est:.3f}s</strong>
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                            
                            with col2:
                                if top_speed_stats:
                                    avg_top = top_speed_stats.get('avg_top_speed', 0)
                                    max_top = top_speed_stats.get('max_top_speed', 0)
                                    st.markdown(f"""
                                    <div style="background: linear-gradient(135deg, rgba(52, 73, 94, 0.95) 0%, rgba(44, 62, 80, 0.95) 100%); 
                                         padding: 1rem 1.25rem; 
                                         margin: 0.5rem 0; 
                                         border-radius: 12px; 
                                         border-left: 4px solid rgba(0, 191, 255, 0.6);
                                         border-top: 1px solid rgba(255, 255, 255, 0.1);
                                         border-right: 1px solid rgba(255, 255, 255, 0.1);
                                         border-bottom: 1px solid rgba(255, 255, 255, 0.1);
                                         box-shadow: 0 4px 16px rgba(0, 0, 0, 0.5);">
                                        <div style="font-size: 0.75rem; 
                                             font-weight: 700; 
                                             color: #00BFFF; 
                                             margin-bottom: 0.5rem; 
                                             letter-spacing: 1px;
                                             text-transform: uppercase;">
                                            Top Speed
                                        </div>
                                        <div style="font-size: 1.4rem; font-weight: 700; color: #FFFFFF; margin-bottom: 0.5rem;">
                                            {avg_top:.1f} km/h
                                        </div>
                                        <div style="font-size: 0.85rem; color: #E8E8E8; line-height: 1.6;">
                                            Average: <strong style="color: #FFFFFF; font-weight: 700;">{avg_top:.1f} km/h</strong><br/>
                                            Maximum: <strong style="color: #FFFFFF; font-weight: 700;">{max_top:.1f} km/h</strong>
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                    
                    # Total Potential Improvement (styled like screenshot)
                    # Calculate from timing points (all 4 segments)
                    total_gain = sum(float(pt.get('potential_gain', 0)) for pt in timing_points.values() if isinstance(pt, dict))
                    
                    # Always show the section even if total_gain is 0
                    if True:
                        # Get current best lap for calculation
                        current_best = driver_sector.get('lap_best', 100.0)
                        projected_best = current_best - total_gain
                        improvement_pct = (total_gain / current_best) * 100 if current_best > 0 else 0
                        
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, rgba(52, 73, 94, 0.95) 0%, rgba(44, 62, 80, 0.95) 100%); 
                             padding: 1rem 1.25rem 2rem; 
                             border-radius: 12px; 
                             border-left: 4px solid rgba(0, 191, 255, 0.6);
                             border-top: 1px solid rgba(255, 255, 255, 0.1);
                             border-right: 1px solid rgba(255, 255, 255, 0.1);
                             border-bottom: 1px solid rgba(255, 255, 255, 0.1);
                             box-shadow: 0 4px 16px rgba(0, 0, 0, 0.5);
                             margin: 0.5rem 0;">
                            <div style="font-size: 0.75rem; font-weight: 700; color: #00BFFF; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 1rem;">
                                Total Potential Improvement
                            </div>
                            <div style="font-size: 0.85rem; color: #E8E8E8; margin-bottom: 1rem;">
                                Implementing all actions could improve lap time by <strong style="color: #FFFFFF; font-weight: 700;">{total_gain:.3f}s</strong>
                            </div>
                            <div style="display: flex; justify-content: space-between; align-items: center; background: rgba(26, 37, 47, 0.6); padding: 1.25rem; border-radius: 12px;">
                                <div>
                                    <div style="font-size: 0.75rem; color: #B0B0B0; font-weight: 600; margin-bottom: 0.25rem;">Current Best Lap</div>
                                    <div style="font-size: 1.4rem; font-weight: 700; color: #FFFFFF;">{current_best:.3f}s</div>
                                </div>
                                <div style="font-size: 2rem; color: #00BFFF; font-weight: 700;">→</div>
                                <div>
                                    <div style="font-size: 0.75rem; color: #B0B0B0; font-weight: 600; margin-bottom: 0.25rem;">Projected Best Lap</div>
                                    <div style="font-size: 1.4rem; font-weight: 700; color: #FFFFFF;">{projected_best:.3f}s</div>
                                </div>
                                <div style="text-align: right;">
                                    <div style="font-size: 0.75rem; color: #B0B0B0; font-weight: 600; margin-bottom: 0.25rem;">Improvement</div>
                                    <div style="font-size: 1.4rem; font-weight: 700; color: #FFFFFF;">{improvement_pct:.2f}%</div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)


                    # Wind & Top Speed Analysis (expander)




        else:
            st.markdown(f"""
            <div style="background: rgba(44, 62, 80, 0.95); 
                 border-left: 6px solid rgba(0, 191, 255, 0.6); 
                 padding: 1rem 1.25rem; 
                 margin: 1rem 0; 
                 border-radius: 16px;">
                <div style="font-size: 1.05rem; font-weight: 600; color: #FFFFFF; margin-bottom: 0.5rem;">
                    No Sector Data Available
                </div>
                <div style="font-size: 0.9rem; color: #E8E8E8;">
                    Sector timing data not available for {selected_driver_name}. This driver may not have completed enough laps with sector timing enabled during the race.
                </div>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="background: rgba(44, 62, 80, 0.95); padding:1rem; border-radius:12px;">
            <div style="font-weight:700;color:#FFFFFF;">Sector Analysis Data Missing</div>
            <div style="color:#E8E8E8;margin-top:0.5rem;">Please run sector analysis to generate the required CSVs, or confirm that the path <code>../outputs/sector_analysis/</code> exists.</div>
        </div>
        """, unsafe_allow_html=True)

    # Footer (unchanged)
    st.markdown("""
    <div style="text-align: center; color: #B0B0B0; padding: 2rem 0; font-size: 0.95rem;">
        <p style="margin: 0; color: #FFFFFF; font-weight: 600;">
            Lap-By-Lap Insights <span style="color: #00BFFF;">|</span> Telemetry-Driven Analytics <span style="color: #00BFFF;">|</span> RaceSense AI
        </p>
    </div>
    """, unsafe_allow_html=True)
