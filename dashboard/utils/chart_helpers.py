"""Helper functions for creating consistent charts across all tabs."""
import plotly.graph_objects as go


def create_standard_bar_chart(x_data, y_data, colors, labels=None, title="", xaxis_title="", yaxis_title="", height=500):
    """Create a standardized bar chart with consistent styling."""
    fig = go.Figure()
    
    if isinstance(colors, str):
        colors = [colors] * len(x_data)
    
    fig.add_trace(go.Bar(
        x=x_data,
        y=y_data,
        marker=dict(
            color=colors,
            line=dict(color='rgba(0, 0, 0, 0.3)', width=2),
            opacity=0.9
        ),
        text=[f"<b>{v:.3f}s</b>" if isinstance(v, float) else f"<b>{v}</b>" for v in y_data],
        textposition='outside',
        textfont=dict(size=11, color='#FFFFFF', family='Arial, sans-serif'),
        hovertemplate='<b>%{x}</b><br>Value: <b>%{y}</b><extra></extra>',
        width=0.6,
        showlegend=False
    ))
    
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(size=17, color='#FFFFFF', family='Arial, sans-serif'),
            x=0.5,
            xanchor='center',
            y=0.98,
            yanchor='top'
        ),
        xaxis_title=dict(text=f"<b>{xaxis_title}</b>", font=dict(size=11, color='#FFFFFF', family='Arial, sans-serif')),
        yaxis_title=dict(text=f"<b>{yaxis_title}</b>", font=dict(size=11, color='#FFFFFF', family='Arial, sans-serif')),
        height=height,
        plot_bgcolor='rgba(26, 37, 47, 0.5)',
        paper_bgcolor='rgba(44, 62, 80, 0.3)',
        font=dict(color='#FFFFFF', size=14, family='Arial, sans-serif'),
        margin=dict(t=100, b=80, l=90, r=60),
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
            linecolor='rgba(0, 191, 255, 0.5)'
        ),
        autosize=True
    )
    
    return fig


def create_grouped_bar_chart(x_data, y_data_dict, colors_dict, title="", xaxis_title="", yaxis_title="", height=650):
    """Create a grouped bar chart with multiple series."""
    fig = go.Figure()
    
    for name, y_data in y_data_dict.items():
        fig.add_trace(go.Bar(
            name=name,
            x=x_data,
            y=y_data,
            marker=dict(
                color=colors_dict.get(name, '#00BFFF'),
                line=dict(color='rgba(0, 0, 0, 0.3)', width=2),
                opacity=0.9
            ),
            text=[f"<b>{v:.3f}s</b>" if isinstance(v, float) else f"<b>{v}</b>" for v in y_data],
            textposition='outside',
            textfont=dict(size=9, color='#FFFFFF', family='Arial, sans-serif'),
            hovertemplate=f'<b>%{{x}}</b><br>{name}: <b>%{{y}}</b><extra></extra>',
            width=0.4
        ))
    
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(size=17, color='#FFFFFF', family='Arial, sans-serif'),
            x=0.5,
            xanchor='center',
            y=0.96,
            yanchor='top'
        ),
        xaxis_title=dict(text=f"<b>{xaxis_title}</b>", font=dict(size=11, color='#FFFFFF', family='Arial, sans-serif')),
        yaxis_title=dict(text=f"<b>{yaxis_title}</b>", font=dict(size=11, color='#FFFFFF', family='Arial, sans-serif')),
        height=height,
        plot_bgcolor='rgba(26, 37, 47, 0.5)',
        paper_bgcolor='rgba(44, 62, 80, 0.3)',
        font=dict(color='#FFFFFF', size=14, family='Arial, sans-serif'),
        margin=dict(t=120, b=80, l=90, r=60),
        barmode='group',
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.12,
            xanchor="center",
            x=0.5,
            font=dict(size=11, color='#FFFFFF', family='Arial, sans-serif'),
            bgcolor='rgba(44, 62, 80, 0.9)',
            bordercolor='rgba(0, 191, 255, 0.5)',
            borderwidth=2
        ),
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
            linecolor='rgba(0, 191, 255, 0.5)'
        ),
        autosize=True
    )
    
    return fig


def add_chart_controls(key_suffix="chart"):
    """Add consistent chart controls toggle."""
    import streamlit as st
    return st.checkbox("Chart Controls", value=False, key=f"tools_{key_suffix}", 
                      help="Show chart controls (zoom, pan, download)")


def get_chart_config(show_tools, RESPONSIVE_CONFIG):
    """Get standardized chart configuration."""
    chart_config = RESPONSIVE_CONFIG['config'].copy()
    chart_config['displayModeBar'] = show_tools
    return chart_config
