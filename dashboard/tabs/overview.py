"""Overview tab - Driver Performance Overview."""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go


def render(selected_driver, selected_driver_name, enhanced_mode, df, COLORS, apply_standard_layout, RESPONSIVE_CONFIG, predictions, commentary, clusters, model_selection, driver_commentary, driver_clusters):
    """Render the Overview tab content."""
    
    # Warning for 0% confidence (enhanced mode only)
    if enhanced_mode and selected_driver.get('confidence_score', 0) == 0:
        st.markdown("""
        <div style="background: rgba(44, 62, 80, 0.3); border-left: 6px solid #FFC107; padding: 1rem 1.25rem; 
             margin: 0.75rem 0 1rem 0; border-radius: 16px; border: 1px solid rgba(255, 193, 7, 0.3);">
            <div style="display: flex; align-items: center; gap: 1rem;">
                <span style="font-size: 2rem;">⚠️</span>
                <div>
                    <strong style="color: #FFFFFF; font-size: 1.1rem;">Low Prediction Reliability</strong>
                    <p style="color: #FFFFFF; margin: 0.5rem 0 0 0; font-size: 0.95rem;">
                        The ML models struggled to accurately predict this driver's performance. 
                        This may be due to inconsistent lap times, limited data, or unique driving patterns. 
                        <strong>Use predictions with caution.</strong>
                    </p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Key metrics - Remove duplicates (Predicted Next and Confidence moved to Predictions tab)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Best Lap",
            f"{selected_driver['best_lap']:.3f}s",
            help="Fastest lap time achieved by this driver"
        )
    
    with col2:
        st.metric(
            "Average Lap",
            f"{selected_driver['avg_lap']:.3f}s",
            help="Average lap time across all laps"
        )
    
    with col3:
        improvement = selected_driver['improvement_vs_best']
        arrow = "↓" if improvement < 0 else "↑"
        st.metric(
            "Gap to Best",
            f"{arrow} {abs(improvement):.3f}s",
            help="Difference from best lap (↓ faster, ↑ slower)"
        )
    
    # Standard spacing: 12px between sections
    st.markdown('<div style="margin: 0.4rem 0;"></div>', unsafe_allow_html=True)
    
    # Performance Comparison
    st.markdown("### Performance Comparison")
    
    fig = go.Figure()
    
    lap_types = ['Best Lap', 'Average Lap']
    lap_values = [
        selected_driver['best_lap'],
        selected_driver['avg_lap']
    ]
    # Clean 2-color palette: Gold for best, Cyan for average
    colors = ['#FFD700', '#00BFFF']
    
    for i, (lap_type, value, color) in enumerate(zip(lap_types, lap_values, colors)):
        fig.add_trace(go.Bar(
            x=[lap_type],
            y=[value],
            marker=dict(
                color=color,
                line=dict(color='rgba(0, 0, 0, 0.3)', width=2),
                opacity=0.9
            ),
            text=f"<b>{value:.3f}s</b>",
            textposition='outside',
            textfont=dict(size=12, color='#FFFFFF', family='Arial, sans-serif'),
            hovertemplate=f'<b>{lap_type}</b><br>Time: <b>{value:.3f}s</b><extra></extra>',
            showlegend=False,
            width=0.6
        ))
    
    # Calculate dynamic range
    min_value = min(lap_values)
    max_value = max(lap_values)
    value_range = max_value - min_value
    y_min = min_value - (value_range * 0.1 if value_range > 0 else min_value * 0.02)
    y_max = max_value + (value_range * 0.2 if value_range > 0 else max_value * 0.05)
    
    fig.update_layout(
        title=dict(
            text=f"<b>{selected_driver_name}</b> - Driver Lap Comparison",
            font=dict(size=20, color='#FFFFFF', family='Arial, sans-serif'),
            x=0.5,
            xanchor='center',
            y=0.98,
            yanchor='top'
        ),
        xaxis_title=dict(text="<b>Lap Type</b>", font=dict(size=12, color='#FFFFFF', family='Arial, sans-serif')),
        yaxis_title=dict(text="<b>Lap Time (seconds)</b>", font=dict(size=12, color='#FFFFFF', family='Arial, sans-serif')),
        height=500,
        plot_bgcolor='rgba(26, 37, 47, 0.5)',
        paper_bgcolor='rgba(44, 62, 80, 0.3)',
        font=dict(color='#FFFFFF', size=14, family='Arial, sans-serif'),
        margin=dict(t=120, b=80, l=90, r=60),
        xaxis=dict(
            showgrid=False,
            tickfont=dict(size=11, color='#FFFFFF', family='Arial, sans-serif'),
            showline=True,
            linewidth=2,
            linecolor='rgba(0, 191, 255, 0.5)'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(255, 255, 255, 0.1)',
            gridwidth=1,
            tickfont=dict(size=11, color='#FFFFFF', family='Arial, sans-serif'),
            showline=True,
            linewidth=2,
            linecolor='rgba(0, 191, 255, 0.5)',
            range=[y_min, y_max]
        ),
        autosize=True
    )
    
    # Chart controls always visible
    chart_config = {
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'performance_comparison',
            'height': 1080,
            'width': 1920,
            'scale': 2
        },
        'responsive': True
    }
    
    st.plotly_chart(fig, width="stretch", config=chart_config, key="overview_chart")
    
    # Standard spacing
    st.markdown('<div style="margin: 0.4rem 0;"></div>', unsafe_allow_html=True)
    
    # Prediction Quality Dashboard (if enhanced mode)
    if enhanced_mode and 'selected_model' in selected_driver:
        st.markdown("### Prediction Quality Dashboard")
        
        conf_score = selected_driver.get('confidence_score', 0)
        model_agreement = selected_driver.get('model_agreement_score', 0)
        models_agree = selected_driver.get('models_in_agreement', 0)
        
        # Reliability rating
        if conf_score >= 80:
            reliability_text = "Excellent"
        elif conf_score >= 60:
            reliability_text = "Good"
        elif conf_score >= 40:
            reliability_text = "Fair"
        elif conf_score >= 20:
            reliability_text = "Poor"
        else:
            reliability_text = "Very Low"
        
        # Risk level
        if conf_score >= 70 and models_agree >= 4:
            risk_level = "Low Risk"
            risk_color = "#28A745"
        elif conf_score >= 50 and models_agree >= 3:
            risk_level = "Moderate Risk"
            risk_color = "#FFD700"
        else:
            risk_level = "High Risk"
            risk_color = "#DC3545"
        
        # Model agreement status
        if models_agree >= 5:
            agreement_status = "Strong Agreement"
            agreement_color = "#28A745"
        elif models_agree >= 4:
            agreement_status = "Good Agreement"
            agreement_color = "#17A2B8"
        elif models_agree >= 3:
            agreement_status = "Moderate Agreement"
            agreement_color = "#FFC107"
        else:
            agreement_status = "Low Agreement"
            agreement_color = "#DC3545"
        
        # Display unified quality dashboard
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(44, 62, 80, 0.6) 0%, rgba(26, 37, 47, 0.6) 100%); 
             border: 2px solid rgba(0, 191, 255, 0.3); border-radius: 16px; padding: 1rem; margin: 0.75rem 0;">
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1.5rem;">
                <div>
                    <div style="font-size: 0.9rem; color: #B0B0B0; font-weight: 600; margin-bottom: 0.5rem;">RELIABILITY</div>
                    <div style="font-size: 1.8rem; font-weight: 900; color: #FFFFFF;">{conf_score:.0f}%</div>
                    <div style="font-size: 0.95rem; color: #B0B0B0; margin-top: 0.3rem;">{reliability_text}</div>
                </div>
                <div>
                    <div style="font-size: 0.9rem; color: #B0B0B0; font-weight: 600; margin-bottom: 0.5rem;">TYPICAL ERROR</div>
                    <div style="font-size: 1.8rem; font-weight: 900; color: #FFFFFF;">±{selected_driver.get('mae', 0):.2f}s</div>
                    <div style="font-size: 0.95rem; color: #B0B0B0; margin-top: 0.3rem;">Average prediction error</div>
                </div>
                <div>
                    <div style="font-size: 0.9rem; color: #B0B0B0; font-weight: 600; margin-bottom: 0.5rem;">MODEL USED</div>
                    <div style="font-size: 1.8rem; font-weight: 900; color: #00BFFF;">{selected_driver.get('selected_model', 'N/A')}</div>
                    <div style="font-size: 0.95rem; color: #B0B0B0; margin-top: 0.3rem;">Best performing algorithm</div>
                </div>
                <div>
                    <div style="font-size: 0.9rem; color: #B0B0B0; font-weight: 600; margin-bottom: 0.5rem;">RISK LEVEL</div>
                    <div style="font-size: 1.8rem; font-weight: 900; color: {risk_color};">{risk_level}</div>
                    <div style="font-size: 0.95rem; color: #B0B0B0; margin-top: 0.3rem;">Prediction uncertainty</div>
                </div>
            </div>
            <hr style="margin: 0.75rem 0; border: none; height: 1px; background: #DEE2E6;">
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem;">
                <div style="text-align: center;">
                    <div style="font-size: 0.85rem; color: #B0B0B0; font-weight: 600;">Model Agreement</div>
                    <div style="font-size: 1.5rem; font-weight: 900; color: {agreement_color}; margin-top: 0.3rem;">{models_agree}/6</div>
                    <div style="font-size: 0.85rem; color: {agreement_color}; margin-top: 0.2rem;">{agreement_status}</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 0.85rem; color: #B0B0B0; font-weight: 600;">Prediction Accuracy</div>
                    <div style="font-size: 1.5rem; font-weight: 900; color: #FFFFFF; margin-top: 0.3rem;">{selected_driver.get('coverage', 0)*100:.0f}%</div>
                    <div style="font-size: 0.85rem; color: #B0B0B0; margin-top: 0.2rem;">of times in range (target: 90%)</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 0.85rem; color: #B0B0B0; font-weight: 600;">Interval Width</div>
                    <div style="font-size: 1.5rem; font-weight: 900; color: #FFFFFF; margin-top: 0.3rem;">{selected_driver.get('interval_width', 0):.1f}s</div>
                    <div style="font-size: 0.85rem; color: #B0B0B0; margin-top: 0.2rem;">Prediction range span</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Standard spacing
        st.markdown('<div style="margin: 0.4rem 0;"></div>', unsafe_allow_html=True)
    
    # Quick Stats
    st.markdown("### Quick Stats")
    
    col_stat1, col_stat2, col_stat3 = st.columns(3)
    
    with col_stat1:
        if enhanced_mode and 'confidence_interval' in selected_driver:
            st.markdown(f"""
            <div class="insight-box" style="min-height: 200px; display: flex; align-items: center; justify-content: center;">
                <div style="text-align: center; width: 100%; padding: 0 0.5rem;">
                    <div style="margin-bottom: 1rem;">
                        <strong style="font-size: 1.2rem; color: #00BFFF; display: block;">Prediction Range</strong>
                    </div>
                    <div style="margin-bottom: 1rem;">
                        <div style="font-size: 1.6rem; font-weight: 900; color: #FFFFFF; line-height: 1.3;">
                            {selected_driver['prediction_lower']:.3f}s - {selected_driver['prediction_upper']:.3f}s
                        </div>
                    </div>
                    <div>
                        <div style="font-size: 0.95rem; color: #B0B0B0; font-weight: 600;">90% Confidence Interval</div>
                        <div style="font-size: 1.3rem; font-weight: 800; color: #FFFFFF; margin-top: 0.3rem;">±{selected_driver['confidence_interval']:.3f}s</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="insight-box" style="min-height: 200px; display: flex; align-items: center; justify-content: center;">
                <div style="text-align: center; width: 100%; padding: 0 0.5rem;">
                    <div style="margin-bottom: 1rem;">
                        <strong style="font-size: 1.2rem; color: #FFD700; display: block;">Average Lap</strong>
                    </div>
                    <div style="margin-bottom: 1rem;">
                        <div style="font-size: 3rem; font-weight: 900; color: #FFD700; line-height: 1;">
                            {selected_driver['avg_lap']:.3f}s
                        </div>
                    </div>
                    <div>
                        <div style="font-size: 0.95rem; color: #B0B0B0; font-weight: 600;">Mean Performance</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with col_stat2:
        # Field Ranking - equal height, consistent sizing
        driver_rank = predictions[predictions['driver_name'] == selected_driver_name].index[0] + 1
        st.markdown(f"""
        <div class="insight-box" style="min-height: 200px; max-height: 200px; display: flex; align-items: center; justify-content: center;">
            <div style="text-align: center; width: 100%; padding: 0 0.5rem;">
                <div style="margin-bottom: 1rem;">
                    <strong style="font-size: 1.2rem; color: #FFD700; display: block;">Field Ranking</strong>
                </div>
                <div style="margin-bottom: 1rem;">
                    <div style="font-size: 4rem; color: #FFD700; font-weight: 900; line-height: 1;">#{driver_rank}</div>
                </div>
                <div>
                    <div style="font-size: 1.1rem; color: #FFFFFF; font-weight: 700;">of {len(predictions)} drivers</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_stat3:
        if enhanced_mode and 'model_agreement_score' in selected_driver:
            models_agree = selected_driver.get('models_in_agreement', 0)
            
            if models_agree >= 5:
                agreement_color = '#28A745'
            elif models_agree >= 4:
                agreement_color = '#17A2B8'
            elif models_agree >= 3:
                agreement_color = '#FFC107'
            else:
                agreement_color = '#DC3545'
            
            st.markdown(f"""
            <div class="insight-box" style="min-height: 200px; display: flex; align-items: center; justify-content: center;">
                <div style="text-align: center; width: 100%; padding: 0 0.5rem;">
                    <div style="margin-bottom: 1rem;">
                        <strong style="font-size: 1.2rem; color: #00BFFF; display: block;">Model Agreement</strong>
                    </div>
                    <div style="margin-bottom: 1rem;">
                        <div style="font-size: 4rem; color: {agreement_color}; font-weight: 900; line-height: 1;">{models_agree}/6</div>
                    </div>
                    <div>
                        <div style="font-size: 1.1rem; color: #FFFFFF; font-weight: 700;">Models Agree</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            consistency_pct = (1 - abs(improvement) / selected_driver['best_lap']) * 100
            consistency_pct = min(100, max(0, consistency_pct))
            
            if consistency_pct >= 95:
                consistency_color = '#28A745'
            elif consistency_pct >= 85:
                consistency_color = '#FFC107'
            else:
                consistency_color = '#DC3545'
            
            st.markdown(f"""
            <div class="insight-box" style="min-height: 200px; display: flex; align-items: center; justify-content: center;">
                <div style="text-align: center; width: 100%; padding: 0 0.5rem;">
                    <div style="margin-bottom: 1rem;">
                        <strong style="font-size: 1.2rem; color: #00BFFF; display: block;">Consistency Score</strong>
                    </div>
                    <div style="margin-bottom: 1rem;">
                        <div style="font-size: 4rem; color: {consistency_color}; font-weight: 900; line-height: 1;">{consistency_pct:.1f}%</div>
                    </div>
                    <div>
                        <div style="font-size: 1.1rem; color: #FFFFFF; font-weight: 700;">vs Best Lap</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
