"""
RaceSenseAI - Toyota GR Cup Race Analytics Dashboard
Entry point for Streamlit Cloud deployment
"""

import sys
import os

# Add dashboard directory to Python path WITHOUT changing working directory
current_dir = os.path.dirname(os.path.abspath(__file__))
dashboard_dir = os.path.join(current_dir, 'dashboard')

# Add both directories to path
if dashboard_dir not in sys.path:
    sys.path.insert(0, dashboard_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import and execute dashboard code in current namespace
dashboard_file = os.path.join(dashboard_dir, 'dashboard.py')
with open(dashboard_file, 'r', encoding='utf-8') as f:
    dashboard_code = f.read()

# Execute in current namespace with proper globals
exec(dashboard_code, {'__file__': dashboard_file, '__name__': '__main__'})
