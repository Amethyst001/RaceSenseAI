"""
Dashboard Configuration
Colors, themes, and constants
"""

# Color Palette
COLORS = {
    'primary': '#EB0A1E',
    'gold': '#FFD700',
    'silver': '#C0C0C0',
    'bronze': '#CD7F32',
    'cyan': '#00BFFF',
    'success': '#28A745',
    'warning': '#FFC107',
    'danger': '#DC3545',
    'white': '#FFFFFF',
    'gray': '#B0B0B0'
}

# Chart Configuration
RESPONSIVE_CONFIG = {
    'config': {
        'displayModeBar': 'hover',
        'displaylogo': False,
        'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'toyota_race_chart',
            'height': 1080,
            'width': 1920,
            'scale': 2
        },
        'responsive': True
    }
}

# Text Configuration
TEXT_CONFIG = {
    'font_family': 'Arial, sans-serif',
    'title_size': 16,
    'axis_title_size': 13,
    'tick_size': 11,
    'legend_size': 11
}

# Hover Template
HOVER_TEMPLATE = '<b>%{x}</b><br>%{y:.3f}s<extra></extra>'

# Page Configuration
PAGE_CONFIG = {
    'page_title': "Toyota GR Cup - Race Analytics",
    'page_icon': "🏁",
    'layout': "wide",
    'initial_sidebar_state': "expanded"
}
