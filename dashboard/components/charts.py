"""
Chart Generation Functions
Reusable chart creation with consistent styling
"""

import plotly.graph_objects as go
from dashboard.config import COLORS, TEXT_CONFIG


def apply_standard_layout(fig, title, xaxis_title, yaxis_title, height=600):
    """Apply standard layout to all charts"""
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(
                size=TEXT_CONFIG['title_size'],
                color=COLORS['white'],
                family=TEXT_CONFIG['font_family']
            ),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title=dict(
                text=xaxis_title,
                font=dict(
                    size=TEXT_CONFIG['axis_title_size'],
                    color=COLORS['white'],
                    family=TEXT_CONFIG['font_family']
                )
            ),
            tickfont=dict(
                size=TEXT_CONFIG['tick_size'],
                color=COLORS['white'],
                family=TEXT_CONFIG['font_family']
            ),
            gridcolor='rgba(255, 255, 255, 0.1)',
            showgrid=True,
            zeroline=False
        ),
        yaxis=dict(
            title=dict(
                text=yaxis_title,
                font=dict(
                    size=TEXT_CONFIG['axis_title_size'],
                    color=COLORS['white'],
                    family=TEXT_CONFIG['font_family']
                )
            ),
            tickfont=dict(
                size=TEXT_CONFIG['tick_size'],
                color=COLORS['white'],
                family=TEXT_CONFIG['font_family']
            ),
            gridcolor='rgba(255, 255, 255, 0.1)',
            showgrid=True,
            zeroline=False
        ),
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        font=dict(
            color=COLORS['white'],
            family=TEXT_CONFIG['font_family']
        ),
        height=height,
        hovermode='closest',
        hoverlabel=dict(
            bgcolor='rgba(0, 0, 0, 0.8)',
            font=dict(
                size=12,
                color=COLORS['white'],
                family=TEXT_CONFIG['font_family']
            ),
            bordercolor='rgba(255, 255, 255, 0.3)'
        )
    )
    return fig


def get_bar_config(color=None, opacity=0.9):
    """Get standard bar configuration"""
    if color is None:
        color = COLORS['primary']
    
    return dict(
        color=color,
        line=dict(color='rgba(0, 0, 0, 0.3)', width=1.5),
        opacity=opacity
    )


def create_performance_bar_chart(data, x_col, y_col, title, color=None):
    """Create a standard performance bar chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=data[x_col],
        y=data[y_col],
        marker=get_bar_config(color),
        text=[f"<b>{v:.3f}s</b>" for v in data[y_col]],
        textposition='outside',
        textfont=dict(size=10, color=COLORS['white'], family=TEXT_CONFIG['font_family']),
        hovertemplate='<b>%{x}</b><br>%{y:.3f}s<extra></extra>'
    ))
    
    fig = apply_standard_layout(fig, title, x_col, y_col)
    return fig


def create_comparison_chart(data, categories, values, title):
    """Create a comparison chart with multiple series"""
    fig = go.Figure()
    
    colors = [COLORS['primary'], COLORS['gold'], COLORS['cyan']]
    
    for idx, (cat, val) in enumerate(zip(categories, values)):
        fig.add_trace(go.Bar(
            name=cat,
            x=data.index.astype(str),
            y=data[val],
            marker=get_bar_config(colors[idx % len(colors)]),
            text=[f"<b>{v:.2f}</b>" for v in data[val]],
            textposition='outside',
            textfont=dict(size=9, color=COLORS['white'], family=TEXT_CONFIG['font_family'])
        ))
    
    fig = apply_standard_layout(fig, title, "Driver", "Time (s)")
    fig.update_layout(barmode='group')
    return fig
