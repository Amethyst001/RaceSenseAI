"""
Standard Chart Configuration for Toyota GR Cup Analytics
Consistent styling across all visualizations
"""

# Toyota Brand Colors
TOYOTA_RED = '#EB0A1E'
TOYOTA_BLACK = '#000000'
TOYOTA_WHITE = '#FFFFFF'
TOYOTA_GRAY = '#58595B'
GOLD = '#FFD700'
SILVER = '#C0C0C0'
BRONZE = '#CD7F32'

# Chart Colors
COLORS = {
    'primary': TOYOTA_RED,
    'secondary': '#4169E1',
    'success': '#28A745',
    'warning': '#FFC107',
    'danger': '#DC3545',
    'info': '#17A2B8',
    'gold': GOLD,
    'silver': SILVER,
    'bronze': BRONZE,
    'predicted': '#32CD32',
    'best': GOLD,
    'average': '#4169E1'
}

# Standard Layout Configuration
CHART_LAYOUT = {
    'font': dict(
        family='Arial, sans-serif',
        size=14,
        color='#2C3E50'
    ),
    'title': dict(
        font=dict(size=22, color='#2C3E50', family='Arial Black'),
        x=0.5,
        xanchor='center',
        y=0.98,
        yanchor='top'
    ),
    'plot_bgcolor': '#FFFFFF',
    'paper_bgcolor': '#FFFFFF',
    'margin': dict(t=100, b=100, l=80, r=60),
    'height': 600,
    'xaxis': dict(
        showgrid=False,
        tickfont=dict(size=13, color='#2C3E50'),
        title_font=dict(size=16, color='#2C3E50'),
        linecolor='#E0E0E0',
        linewidth=2
    ),
    'yaxis': dict(
        showgrid=True,
        gridcolor='#E9ECEF',
        gridwidth=1,
        tickfont=dict(size=13, color='#2C3E50'),
        title_font=dict(size=16, color='#2C3E50'),
        linecolor='#E0E0E0',
        linewidth=2
    ),
    'legend': dict(
        orientation='h',
        yanchor='bottom',
        y=1.02,
        xanchor='right',
        x=1,
        font=dict(size=13),
        bgcolor='rgba(255,255,255,0.8)',
        bordercolor='#E0E0E0',
        borderwidth=1
    )
}

# Text Configuration
TEXT_CONFIG = {
    'textfont': dict(size=14, color='#2C3E50', family='Arial'),
    'textposition': 'outside',
    'texttemplate': '%{text:.3f}s'
}

# Hover Template
HOVER_TEMPLATE = '<b>%{x}</b><br>Value: %{y:.3f}s<extra></extra>'

def get_bar_config(width=0.6):
    """Get standard bar chart configuration"""
    return {
        'width': width,
        'marker': dict(
            line=dict(color='#FFFFFF', width=1.5)
        )
    }

def get_line_config():
    """Get standard line chart configuration"""
    return {
        'mode': 'lines+markers',
        'line': dict(width=3),
        'marker': dict(size=8, line=dict(width=2, color='#FFFFFF'))
    }

def apply_standard_layout(fig, title, xaxis_title, yaxis_title, height=600):
    """Apply standard layout to any figure"""
    layout = CHART_LAYOUT.copy()
    layout['title']['text'] = title
    layout['xaxis']['title'] = xaxis_title
    layout['yaxis']['title'] = yaxis_title
    layout['height'] = height
    fig.update_layout(**layout)
    return fig
