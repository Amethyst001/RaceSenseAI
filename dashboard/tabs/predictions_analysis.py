"""Predictions & Analysis Tab - Merged predictions with AI commentary."""
import streamlit as st
import plotly.graph_objects as go
import re


def render(selected_driver, selected_driver_name, enhanced_mode, df, COLORS, apply_standard_layout, 
           RESPONSIVE_CONFIG, predictions, commentary, clusters, model_selection, driver_commentary, driver_clusters):
    """Render merged Predictions & Analysis tab"""
    
    # Warning for 0% confidence (enhanced mode only)
    if enhanced_mode and selected_driver.get('confidence_score', 0) == 0:
        st.markdown("""
        <div style="background: rgba(44, 62, 80, 0.3); border-left: 6px solid #FFC107; padding: 1rem 1.25rem; 
             margin: 1rem 0; border-radius: 16px; border: 1px solid rgba(255, 193, 7, 0.3);">
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
    
    # Key metrics - Remove Gap to Best duplicate (now only in Driver Overview)
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "Predicted Next Lap",
            f"{selected_driver['predicted_next_lap']:.3f}s",
            help="ML-predicted next lap time"
        )
    
    with col2:
        if enhanced_mode:
            confidence_score = selected_driver.get('confidence_score', 0)
            st.metric(
                "Prediction Confidence",
                f"{confidence_score:.0f}%",
                help="ML model confidence score (0-100%)"
            )
        else:
            confidence = selected_driver.get('confidence', 'N/A')
            st.metric(
                "Prediction Confidence",
                f"{confidence}",
                help="Prediction confidence level"
            )
    
    # Standard spacing: 12px between sections
    st.markdown('<div style="margin: 0.4rem 0;"></div>', unsafe_allow_html=True)
    
    # Model Distribution (moved from sidebar)
    if enhanced_mode and model_selection is not None:
        #st.markdown('<hr class="divider-subtle">', unsafe_allow_html=True)
        st.markdown("### Model Distribution")
        
        model_dist = model_selection['model'].value_counts()
        st.markdown(f"**{len(model_selection)} drivers analyzed across 6 ML models**")
        
        # All 5 models in one row with equal smaller sizes
        model_items = list(model_dist.items())
        cols = st.columns(5)
        
        for i, (model, count) in enumerate(model_items):
            driver_text = "driver" if count == 1 else "drivers"
            with cols[i]:
                st.metric(f"{model}", f"{count} {driver_text}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Divider
    #st.markdown('<hr class="divider-subtle">', unsafe_allow_html=True)
    
    # Prediction Range Visualization - Modern Clean Design
    st.markdown("### Prediction Range")
    
    pred_lower = selected_driver.get('prediction_lower', selected_driver['predicted_next_lap'] - 1)
    pred_upper = selected_driver.get('prediction_upper', selected_driver['predicted_next_lap'] + 1)
    best_lap = selected_driver['best_lap']
    predicted = selected_driver['predicted_next_lap']
    
    fig = go.Figure()
    
    # Add thin horizontal line for range
    fig.add_trace(go.Scatter(
        x=[pred_lower, pred_upper],
        y=[1, 1],
        mode='lines',
        line=dict(color='rgba(255, 193, 7, 0.4)', width=3),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Add range interval markers (circles with borders)
    fig.add_trace(go.Scatter(
        x=[pred_lower, pred_upper],
        y=[1, 1],
        mode='markers',
        marker=dict(
            size=16,
            color='rgba(255, 193, 7, 0.3)',
            line=dict(color='#FFC107', width=3)
        ),
        name='90% Range',
        hovertemplate='<b>Range Bound</b><br>%{x:.3f}s<extra></extra>'
    ))
    
    # Add predicted point (filled circle) - no text
    fig.add_trace(go.Scatter(
        x=[predicted],
        y=[1],
        mode='markers',
        marker=dict(
            size=22,
            color='#00BFFF',
            line=dict(color='#FFFFFF', width=3)
        ),
        text=None,
        textposition=None,
        name='Predicted',
        hovertemplate=f'<b>Predicted</b><br>{predicted:.3f}s<extra></extra>'
    ))
    
    # Add best lap marker (filled circle) - no text
    fig.add_trace(go.Scatter(
        x=[best_lap],
        y=[1],
        mode='markers',
        marker=dict(
            size=20,
            color='#28A745',
            line=dict(color='#FFFFFF', width=3)
        ),
        text=None,
        textposition=None,
        name='Best Lap',
        hovertemplate=f'<b>Best Lap</b><br>{best_lap:.3f}s<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text=f"<b>{selected_driver_name}</b> - 90% Confidence Interval",
            font=dict(size=20, color='#FFFFFF', family='Arial, sans-serif'),
            x=0.5,
            xanchor='center',
            y=0.98,
            yanchor='top'
        ),
        xaxis_title=dict(text="<b>Lap Time (seconds)</b>", font=dict(size=11, color='#FFFFFF', family='Arial, sans-serif')),
        yaxis=dict(visible=False, range=[0.8, 1.2]),
        height=450,
        plot_bgcolor='rgba(26, 37, 47, 0.5)',
        paper_bgcolor='rgba(44, 62, 80, 0.3)',
        font=dict(color='#FFFFFF', size=14, family='Arial, sans-serif'),
        margin=dict(t=120, b=100, l=60, r=60),
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(255, 255, 255, 0.1)',
            tickfont=dict(size=11, color='#FFFFFF', family='Arial, sans-serif'),
            showline=True,
            linewidth=2,
            linecolor='rgba(0, 191, 255, 0.5)'
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.45,
            xanchor="center",
            x=0.5,
            font=dict(size=11, color='#FFFFFF'),
            bgcolor='rgba(44, 62, 80, 0.9)',
            bordercolor='rgba(0, 191, 255, 0.5)',
            borderwidth=2
        ),
        autosize=True
    )
    
    # Chart controls positioned to not overlap with title
    chart_config = {
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
        'modeBarButtonsToAdd': [],
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'prediction_range',
            'height': 1080,
            'width': 1920,
            'scale': 2
        },
        'responsive': True
    }
    
    st.plotly_chart(fig, width="stretch", config=chart_config, key="prediction_range_chart")
    
    # Prediction details in insight boxes
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="insight-box" style="min-height: 180px;">
            <strong style="font-size: 1.15rem; color: #00BFFF; display: block; margin-bottom: 0.5rem;">Model Information</strong>
            <div style="font-size: 0.95rem; line-height: 1.8; color: #E8E8E8;">
                <strong>Selected Model:</strong> {selected_driver.get('selected_model', 'Unknown')}<br/>
                <strong>Model Agreement:</strong> {selected_driver.get('models_in_agreement', 'N/A')}/6 models<br/>
                <strong>Typical Error:</strong> ±{selected_driver.get('mae', 0):.2f}s
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="insight-box" style="min-height: 180px;">
            <strong style="font-size: 1.15rem; color: #00BFFF; display: block; margin-bottom: 0.5rem;">Prediction Range</strong>
            <div style="font-size: 0.95rem; line-height: 1.8; color: #E8E8E8;">
                <strong>Lower Bound:</strong> {pred_lower:.3f}s<br/>
                <strong>Upper Bound:</strong> {pred_upper:.3f}s<br/>
                <strong>Range Width:</strong> {pred_upper - pred_lower:.3f}s
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # AI Commentary removed - moved to Driving Insights tab
    
    # Skip AI commentary section in predictions tab
    if False and len(driver_commentary) > 0:
        # Remove all icons from commentary text
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            u"\U0001F900-\U0001F9FF"  # supplemental symbols
            u"\U00002600-\U000026FF"  # misc symbols
            "]+", flags=re.UNICODE)
        
        # Define category order - merge Prediction Reliability and Model Insights
        category_order = ['Performance Prediction', 'Model Reliability & Insights', 'Race Strategy', 'Weather Impact']
        
        # Merge Prediction Reliability and Model Insights
        reliability_comments = driver_commentary[driver_commentary['category'].isin(['Prediction Reliability', 'Model Insights'])]
        if len(reliability_comments) > 0:
            # Create merged category
            merged_text = '<br/><br/>'.join([emoji_pattern.sub('', c['commentary']).strip() for _, c in reliability_comments.iterrows()])
            # Add to commentary with new category
            if 'Prediction Reliability' in driver_commentary['category'].values or 'Model Insights' in driver_commentary['category'].values:
                # Remove old categories from display
                driver_commentary = driver_commentary[~driver_commentary['category'].isin(['Prediction Reliability', 'Model Insights'])]
        
        # Get available categories in the desired order
        available_categories = [cat for cat in category_order if cat in driver_commentary['category'].values]
        
        # Display each category with uniform styling
        for category in available_categories:
            category_comments = driver_commentary[driver_commentary['category'] == category]
            
            # Uniform box style for all categories
            box_color = 'rgba(44, 62, 80, 0.95)'
            border_color = '#00BFFF'
            text_color = '#FFFFFF'
            
            # If multiple comments, show as formatted list; otherwise show as single text
            if len(category_comments) > 1:
                content_html = f'<div style="margin-bottom: 1rem;"><strong style="font-size: 1.15rem; color: {border_color}; display: block; margin-bottom: 0.5rem;">{category}</strong></div>'
                for _, comment in category_comments.iterrows():
                    clean_text = emoji_pattern.sub('', comment["commentary"]).strip()
                    content_html += f'<div style="margin: 0.5rem 0; padding-left: 1rem; border-left: 2px solid {border_color}; line-height: 1.6;">{clean_text}</div>'
                
                st.markdown(f"""
                <div style="background: {box_color}; 
                     padding: 1rem 1.25rem; margin: 1rem 0; border-radius: 12px; 
                     border-left: 6px solid {border_color};
                     border-top: 1px solid rgba(255, 255, 255, 0.1);
                     border-right: 1px solid rgba(255, 255, 255, 0.1);
                     border-bottom: 1px solid rgba(255, 255, 255, 0.1);
                     color: {text_color};">
                    {content_html}
                </div>
                """, unsafe_allow_html=True)
            else:
                comment_text = emoji_pattern.sub('', category_comments.iloc[0]['commentary']).strip()
                # Fix common grammar issues and formatting
                comment_text = comment_text.replace('while not all models agree', 'with some model disagreement')
                comment_text = comment_text.replace('While not all models agree', 'With some model disagreement')
                comment_text = comment_text.replace('the majority (6/6)', 'all 6 models')
                comment_text = comment_text.replace('(6/6)', '(all 6 models)')
                comment_text = comment_text.replace('(5/6)', '(5 of 6 models)')
                comment_text = comment_text.replace('(4/6)', '(4 of 6 models)')
                comment_text = comment_text.replace('(3/6)', '(3 of 6 models)')
                # Fix tautology
                comment_text = comment_text.replace('Expected Predicted', 'Expected')
                comment_text = comment_text.replace('Predicted Expected', 'Expected')
                # Fix pipe separators
                comment_text = comment_text.replace(' | ', ' • ')
                # Fix bullet points - add line breaks
                comment_text = comment_text.replace('•', '<br/>• ')
                # Remove extra spaces
                comment_text = comment_text.replace('  ', ' ')
                
                st.markdown(f"""
                <div style="background: {box_color}; 
                     padding: 1rem 1.25rem; margin: 1rem 0; border-radius: 12px; 
                     border-left: 6px solid {border_color};
                     border-top: 1px solid rgba(255, 255, 255, 0.1);
                     border-right: 1px solid rgba(255, 255, 255, 0.1);
                     border-bottom: 1px solid rgba(255, 255, 255, 0.1);
                     color: {text_color};">
                    <div style="margin-bottom: 0.5rem;">
                        <strong style="font-size: 1.15rem; color: {border_color}; display: block;">{category}</strong>
                    </div>
                    <div style="font-size: 0.95rem; line-height: 1.7; color: #E8E8E8;">
                        {comment_text}
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # Model performance display (single selected model)
    if enhanced_mode and model_selection is not None:
        st.markdown('<hr class="divider-glow">', unsafe_allow_html=True)
        st.markdown("### Model Performance")
        
        driver_models = model_selection[model_selection['driver_id'] == selected_driver['driver_id']]
        
        if len(driver_models) > 0:
            selected_model = driver_models.iloc[0]['model']
            mae_value = driver_models.iloc[0]['mae']
            
            fig = go.Figure()
            
            # Compact gauge chart with consistent spacing and colors
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=mae_value,
                domain={'x': [0, 1], 'y': [0, 0.85]},
                number={'suffix': "s", 'font': {'size': 36, 'color': '#FFFFFF'}, 'xanchor': 'center', 'x': 0.5},
                gauge={
                    'axis': {
                        'range': [None, 2.0], 
                        'tickwidth': 1.5, 
                        'tickcolor': "#FFFFFF",
                        'tickfont': {'size': 10, 'color': '#FFFFFF'}
                    },
                    'bar': {'color': "rgba(0, 191, 255, 0.8)", 'thickness': 0.7},
                    'bgcolor': "rgba(26, 37, 47, 0.5)",
                    'borderwidth': 2,
                    'bordercolor': "rgba(0, 191, 255, 0.5)",
                    'steps': [
                        {'range': [0, 0.5], 'color': "rgba(0, 191, 255, 0.2)"},
                        {'range': [0.5, 1.0], 'color': "rgba(0, 191, 255, 0.15)"},
                        {'range': [1.0, 2.0], 'color': "rgba(0, 191, 255, 0.1)"}
                    ],
                    'threshold': {
                        'line': {'color': "rgba(0, 191, 255, 0.8)", 'width': 2.5},
                        'thickness': 0.75,
                        'value': mae_value
                    }
                }
            ))
            
            fig.update_layout(
                title=dict(
                    text=f"<b>{selected_model}</b><br><span style='font-size:12px'>Mean Absolute Error (Lower is Better)</span>",
                    font=dict(size=16, color='#FFFFFF', family='Arial, sans-serif'),
                    x=0.5,
                    xanchor='center',
                    y=0.95,
                    yanchor='top'
                ),
                height=350,
                plot_bgcolor='rgba(26, 37, 47, 0.5)',
                paper_bgcolor='rgba(44, 62, 80, 0.3)',
                font=dict(color='#FFFFFF', family='Arial, sans-serif'),
                margin=dict(t=120, b=50, l=60, r=60),
                autosize=True
            )
            
            # Chart controls always visible
            chart_config = {
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                'responsive': True
            }
            
            st.plotly_chart(fig, width="stretch", config=chart_config, key="model_comparison_chart")
            
            # Success message
            st.markdown(f"""
            <div class="success-box">
            <strong>SELECTED MODEL: {selected_driver.get('selected_model', 'Unknown')}</strong><br/>
            This model achieved the lowest prediction error for this driver.
            </div>
            """, unsafe_allow_html=True)
