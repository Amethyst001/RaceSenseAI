"""Leaderboard tab - Race Leaderboard."""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go


def render(selected_driver, selected_driver_name, enhanced_mode, df, COLORS, apply_standard_layout, RESPONSIVE_CONFIG, predictions, commentary, clusters, model_selection, driver_commentary, driver_clusters):
    """Render the Leaderboard tab content."""
    
    # Start with visualization (chart before table)
    st.markdown("### Top Drivers (Chart)")
    
    top_15 = predictions.head(15).iloc[::-1]  # Reverse for better visual (fastest at top)
    
    fig = go.Figure()
    
    # Single horizontal bar showing predicted times with reduced color intensity
    fig.add_trace(go.Bar(
        y=top_15['driver_name'],
        x=top_15['predicted_next_lap'],
        orientation='h',
        marker=dict(
            color=top_15['predicted_next_lap'],
            colorscale=[
                [0, 'rgba(40, 167, 69, 0.5)'],    # Green with 50% opacity
                [0.5, 'rgba(255, 215, 0, 0.5)'],  # Gold with 50% opacity
                [1, 'rgba(220, 53, 69, 0.5)']     # Red with 50% opacity
            ],
            showscale=False,
            line=dict(color='rgba(0, 0, 0, 0.3)', width=2),
            opacity=1.0
        ),
        text=[f"<b>{v:.3f}s</b>" for v in top_15['predicted_next_lap']],
        textposition='outside',
        textfont=dict(size=11, color='#FFFFFF', family='Arial, sans-serif'),
        hovertemplate='<b>%{y}</b><br>Predicted: <b>%{x:.3f}s</b><extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text="<b>Top 15 Fastest Predicted Lap Times</b>",
            font=dict(size=20, color='#FFFFFF', family='Arial, sans-serif'),
            x=0.5,
            xanchor='center',
            y=0.98,
            yanchor='top'
        ),
        xaxis_title=dict(text="<b>Predicted Lap Time (seconds)</b>", font=dict(size=11, color='#FFFFFF', family='Arial, sans-serif')),
        yaxis_title=dict(text="<b>Driver (Rank Order)</b>", font=dict(size=11, color='#FFFFFF', family='Arial, sans-serif')),
        height=600,
        plot_bgcolor='rgba(26, 37, 47, 0.5)',
        paper_bgcolor='rgba(44, 62, 80, 0.3)',
        font=dict(color='#FFFFFF', size=14, family='Arial, sans-serif'),
        margin=dict(t=120, b=80, l=120, r=80),
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(255, 255, 255, 0.1)',
            tickfont=dict(size=11, color='#FFFFFF', family='Arial, sans-serif'),
            showline=True,
            linewidth=2,
            linecolor='rgba(0, 191, 255, 0.5)'
        ),
        yaxis=dict(
            tickfont=dict(size=11, color='#FFFFFF'),
            showline=True,
            linewidth=2,
            linecolor='rgba(0, 191, 255, 0.5)',
            showgrid=False
        ),
        autosize=True
    )
    
    # Chart controls always visible
    chart_config = {
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
        'responsive': True
    }
    
    st.plotly_chart(fig, width="stretch", config=chart_config, key="leaderboard_chart")
    
    # Add table after chart
    st.markdown('<hr class="divider-subtle">', unsafe_allow_html=True)
    st.markdown("### Driver Ranking")
    
    # Prepare clean table
    if enhanced_mode:
        display_df = predictions[['driver_name', 'predicted_next_lap', 'confidence_score', 
                                   'models_in_agreement', 'selected_model']].copy()
        display_df.columns = ['Driver', 'Predicted (s)', 'Confidence (%)', 'Agreement', 'Model']
        
        display_df['Predicted (s)'] = display_df['Predicted (s)'].apply(lambda x: f"{x:.3f}")
        display_df['Confidence (%)'] = display_df['Confidence (%)'].apply(lambda x: f"{int(x)}")
        display_df['Agreement'] = display_df['Agreement'].astype(str) + '/6'
    else:
        display_df = predictions[['driver_name', 'predicted_next_lap', 'best_lap', 'avg_lap', 'confidence']].copy()
        display_df.columns = ['Driver', 'Predicted (s)', 'Best (s)', 'Avg (s)', 'Confidence']
        
        for col in ['Predicted (s)', 'Best (s)', 'Avg (s)']:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}")
    
    display_df.insert(0, 'Rank', range(1, len(display_df) + 1))
    
    # Display table with better styling
    st.dataframe(
        display_df,
        width="stretch",
        height=500,
        hide_index=True,
        use_container_width=True
    )
