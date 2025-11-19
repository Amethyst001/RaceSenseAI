"""
Toyota GR Cup Indianapolis Race 1 - Interactive Dashboard
Unified dashboard for all race analytics
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import datetime
import importlib
import sys

# Vehicle ID mapping (vehicle_number -> full vehicle_id)
VEHICLE_ID_MAP = {
    '0': 'GR86-002-000', '2': 'GR86-060-2', '3': 'GR86-040-3', '5': 'GR86-065-5',
    '7': 'GR86-006-7', '8': 'GR86-012-8', '11': 'GR86-035-11', '13': 'GR86-022-13',
    '16': 'GR86-010-16', '18': 'GR86-030-18', '21': 'GR86-047-21', '31': 'GR86-015-31',
    '46': 'GR86-033-46', '47': 'GR86-025-47', '55': 'GR86-016-55', '57': 'GR86-057-57',
    '72': 'GR86-026-72', '80': 'GR86-013-80', '86': 'GR86-021-86', '88': 'GR86-049-88',
    '89': 'GR86-028-89', '93': 'GR86-038-93', '98': 'GR86-036-98', '113': 'GR86-063-113',
}

# Driver name mapping (vehicle_id -> driver name)
DRIVER_NAMES = {
    'GR86-002-000': 'Unknown', 'GR86-060-2': 'Will Robusto', 'GR86-040-3': 'Jason Kos',
    'GR86-065-5': 'Beltre Curtis', 'GR86-006-7': 'Jaxon Bell', 'GR86-012-8': 'Tom Rudnai',
    'GR86-035-11': 'Farran Davis', 'GR86-022-13': 'Westin Workman', 'GR86-010-16': 'Unknown',
    'GR86-030-18': 'Rutledge Wood', 'GR86-047-21': 'Michael Edwards', 'GR86-015-31': 'Jackson Tovo',
    'GR86-033-46': 'Lucas Weisenberg', 'GR86-025-47': 'Parker DeLong', 'GR86-016-55': 'Spike Kohlbecker',
    'GR86-057-57': 'Jeff Curry', 'GR86-026-72': 'Ethan Goulart', 'GR86-013-80': 'Paityn Feyen',
    'GR86-021-86': 'Andrew Gilleland', 'GR86-049-88': 'Henry Drury', 'GR86-028-89': 'Livio Galanti',
    'GR86-038-93': 'Patrick Brunson', 'GR86-036-98': 'Max Schweid', 'GR86-063-113': 'Ethan Tovo',
}

# Clear any cached modules - FORCE RELOAD
for module in list(sys.modules.keys()):
    if module.startswith('tabs.') or module.startswith('utils.') or module.startswith('ai_engine.'):
        del sys.modules[module]

# Force reimport with version check
import importlib
DASHBOARD_VERSION = "2.1.0"  # Increment this to force reload

# Import asset loader (now in same folder)
from utils import load_all_assets

# Import tab modules (renamed from pages to avoid Streamlit's multi-page feature)
from tabs import overview, leaderboard, sectors
from tabs import predictions_analysis, insights as insights_tab

# Import standard chart configuration (now in same folder)
try:
    from chart_config import COLORS, apply_standard_layout, get_bar_config, TEXT_CONFIG, HOVER_TEMPLATE
except ImportError:
    # Fallback if chart_config not available
    COLORS = {
        'primary': '#EB0A1E', 'gold': '#FFD700', 'silver': '#C0C0C0',
        'bronze': '#CD7F32', 'predicted': '#00BFFF', 'best': '#FFD700',
        'average': '#4169E1', 'success': '#28A745', 'warning': '#FFC107', 'danger': '#DC3545',
        'cyan': '#00BFFF'
    }
    def apply_standard_layout(fig, title, xaxis_title, yaxis_title, height=600):
        return fig

# Import driver name mapper
try:
    # Add parent directory to path
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from ai_engine.driver_mapping import get_driver_name
    DRIVER_MAPPER_AVAILABLE = True
    print(f"[SUCCESS] Driver mapper loaded successfully")
except Exception as e:
    print(f"[ERROR] Driver mapper not available: {e}")
    import traceback
    traceback.print_exc()
    DRIVER_MAPPER_AVAILABLE = False
    def get_driver_name(vehicle_id):
        return vehicle_id

st.set_page_config(
    page_title="Toyota GR Cup - Race Analytics",
    page_icon="🏁",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo',
        'Report a bug': 'https://github.com/your-repo/issues',
        'About': '# Toyota GR Cup Race Analytics\nAdvanced ML-powered race analysis dashboard'
    }
)

# Detect screen size and adjust accordingly
# This makes charts responsive to different screen sizes
RESPONSIVE_CONFIG = {
    'config': {
        'displayModeBar': 'hover',  # Only show on hover
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

# Load custom CSS and JavaScript from assets
load_all_assets()



# Initialize session state for scroll position management
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.scroll_position = 0

# Load Data
@st.cache_data(ttl=60)  # Cache for 60 seconds, then reload
def load_data():
    try:
        # Get the project root directory (parent of dashboard folder)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Load enhanced multi-model predictions first
        predictions_path = os.path.join(project_root, 'outputs/predictions/per_driver_predictions.csv')
        if os.path.exists(predictions_path):
            predictions = pd.read_csv(predictions_path)
            model_selection = pd.read_csv(os.path.join(project_root, 'models/per_driver_model_selection.csv'))
            enhanced_mode = True
        else:
            # Use per-driver predictions
            predictions = pd.read_csv(os.path.join(project_root, 'outputs/predictions/per_driver_predictions.csv'))
            model_selection = None
            enhanced_mode = False
        
        commentary = pd.read_csv(os.path.join(project_root, 'outputs/commentary/ai_commentary.csv'))
        
        try:
            clusters = pd.read_csv(os.path.join(project_root, 'outputs/analysis/driver_clusters.csv'))
        except:
            clusters = pd.DataFrame()
        
        # Data validation: Basic sanity checks
        # Lap times should be positive and reasonable (30-300 seconds for this track)
        if 'best_lap' in predictions.columns:
            invalid_best = (predictions['best_lap'] <= 0) | (predictions['best_lap'] > 300)
            if invalid_best.any():
                print(f"⚠️ Warning: {invalid_best.sum()} drivers have invalid best_lap values")
        
        if 'avg_lap' in predictions.columns:
            invalid_avg = (predictions['avg_lap'] <= 0) | (predictions['avg_lap'] > 300)
            if invalid_avg.any():
                print(f"⚠️ Warning: {invalid_avg.sum()} drivers have invalid avg_lap values")
        
        return predictions, commentary, clusters, model_selection, enhanced_mode, True
    except Exception as e:
        st.error(f"⚠️ No data found. Please run: py scripts\analyze_race.py\nError: {e}")
        return None, None, None, None, False, False

predictions, commentary, clusters, model_selection, enhanced_mode, data_loaded = load_data()

if not data_loaded:
    st.stop()

# Filter out invalid data (negative lap times, missing data) - silently
if 'best_lap' in predictions.columns:
    valid_mask = (predictions['best_lap'] > 0) & (predictions['best_lap'] < 300)
    if (~valid_mask).any():
        predictions = predictions[valid_mask].reset_index(drop=True)
        # Also filter commentary and clusters
        valid_drivers = predictions['driver_name'].unique()
        commentary = commentary[commentary['driver_name'].isin(valid_drivers)]
        if not clusters.empty:
            clusters = clusters[clusters['driver_name'].isin(valid_drivers)]

# Header with live status and branding
import datetime
current_time = datetime.datetime.now().strftime('%H:%M:%S')

# Logo moved to sidebar - see sidebar section below

st.markdown('<div class="main-header">TOYOTA GR CUP<br/>RACE ANALYTICS DASHBOARD</div>', unsafe_allow_html=True)

# Sidebar with logo at very top
try:
    logo_path = os.path.join(os.path.dirname(__file__), "assets", "racesense_logo.png")
    st.sidebar.markdown("<div style='margin-top: -8rem;'></div>", unsafe_allow_html=True)
    st.sidebar.image(logo_path, width='stretch')
    st.sidebar.markdown("<div style='margin-bottom: 0.5rem;'></div>", unsafe_allow_html=True)
except Exception as e:
    # Fallback if logo not found
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #00BFFF 0%, #0099CC 100%); 
         border-radius: 12px; margin: -5rem 0 0.5rem 0;">
        <div style="color: #FFFFFF; font-size: 1.5rem; font-weight: 900; font-family: Arial Black;">
            RaceSense AI
        </div>
        <div style="color: #FFD700; font-size: 0.8rem; font-weight: 600; letter-spacing: 1px;">
            POWERED BY ML
        </div>
    </div>
    """, unsafe_allow_html=True)

st.sidebar.markdown("### Car Selection")
st.sidebar.markdown('<hr class="divider-subtle">', unsafe_allow_html=True)

# Search functionality with session state
if 'search_query' not in st.session_state:
    st.session_state.search_query = ""
if 'selected_driver_index' not in st.session_state:
    st.session_state.selected_driver_index = 0

search = st.sidebar.text_input("Search Car", st.session_state.search_query, placeholder="Type car number...", key="driver_search")

drivers = sorted(predictions['driver_name'].unique())

# Set default to car 47 (GR86-025-47)
if 'default_set' not in st.session_state:
    default_driver_name = 'GR86-025-47'  # Car 47
    try:
        st.session_state.selected_driver_index = drivers.index(default_driver_name)
    except ValueError:
        # Fallback to best performing driver if car 47 not found
        best_driver_idx = predictions['best_lap'].idxmin()
        best_driver_name = predictions.loc[best_driver_idx, 'driver_name']
        try:
            st.session_state.selected_driver_index = drivers.index(best_driver_name)
        except ValueError:
            st.session_state.selected_driver_index = 0
    st.session_state.default_set = True

# Filter drivers based on search
if search:
    drivers = [d for d in drivers if search.lower() in d.lower()]
    if not drivers:
        st.sidebar.warning("No drivers found")
        drivers = sorted(predictions['driver_name'].unique())

# Ensure index is valid
if st.session_state.selected_driver_index >= len(drivers):
    st.session_state.selected_driver_index = 0

selected_driver_name = st.sidebar.selectbox(
    "Select Car", 
    drivers, 
    index=st.session_state.selected_driver_index,
    key="driver_selector"
)

# Show driver count info (removed as requested)

selected_driver = predictions[predictions['driver_name'] == selected_driver_name].iloc[0]
driver_commentary = commentary[commentary['driver_name'] == selected_driver_name]
driver_clusters = clusters[clusters['driver_name'] == selected_driver_name] if not clusters.empty else pd.DataFrame()

# Display header with driver name after selection
driver_name = DRIVER_NAMES.get(selected_driver_name)
if driver_name and driver_name != 'Unknown':
    st.markdown(f"""
    <div style="text-align: center; margin-bottom: 2rem;">
        <span style="background: #28A745; color: white; padding: 0.75rem 2rem; border-radius: 20px; font-weight: 600; box-shadow: 0 2px 4px rgba(0,0,0,0.2); margin-right: 1rem;">
            Live Data | Last Updated: {current_time}
        </span>
        <span style="background: #00BFFF; color: white; padding: 0.75rem 2rem; border-radius: 20px; font-weight: 600; box-shadow: 0 2px 4px rgba(0,0,0,0.2);">
            Driver: {driver_name}
        </span>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div style="text-align: center; margin-bottom: 2rem;">
        <span style="background: #28A745; color: white; padding: 0.75rem 2rem; border-radius: 20px; font-weight: 600; box-shadow: 0 2px 4px rgba(0,0,0,0.2);">
            Live Data | Last Updated: {current_time}
        </span>
    </div>
    """, unsafe_allow_html=True)

# Model distribution moved to Predictions tab

st.sidebar.markdown('<hr class="divider-subtle">', unsafe_allow_html=True)
st.sidebar.markdown("### Quick Stats")

if enhanced_mode:
    # Enhanced stats with multi-model info
    st.sidebar.markdown(f"""
    <div style="background: linear-gradient(135deg, rgba(44, 62, 80, 0.5) 0%, rgba(26, 37, 47, 0.5) 100%); 
         padding: 1rem; border-radius: 12px; margin: 1rem 0; border: 2px solid rgba(0, 191, 255, 0.3);">
        <div style="margin-bottom: 1.5rem;">
            <div style="color: #FFFFFF; font-size: 0.9rem; font-weight: 600; margin-bottom: 0.5rem;">Next Lap Prediction</div>
            <div style="color: #FFD700; font-size: 1.8rem; font-weight: 900;">{selected_driver['predicted_next_lap']:.3f}s</div>
            <div style="color: #FFD700; font-size: 0.8rem;">± {selected_driver.get('confidence_interval', 0):.3f}s</div>
        </div>
        <div style="margin-bottom: 1.5rem;">
            <div style="color: #FFFFFF; font-size: 0.9rem; font-weight: 600; margin-bottom: 0.5rem;">Driver Best Lap</div>
            <div style="color: #FFD700; font-size: 1.6rem; font-weight: 900;">{selected_driver['best_lap']:.3f}s</div>
        </div>
        <div style="margin-bottom: 1.5rem;">
            <div style="color: #FFFFFF; font-size: 0.9rem; font-weight: 600; margin-bottom: 0.5rem;">Model Used</div>
            <div style="color: #FFD700; font-size: 1.2rem; font-weight: 900;">{selected_driver.get('selected_model', 'N/A')}</div>
        </div>
        <div>
            <div style="color: #FFFFFF; font-size: 0.9rem; font-weight: 600; margin-bottom: 0.5rem;">Prediction Confidence</div>
            <div style="color: #FFD700; font-size: 1.6rem; font-weight: 900;">{selected_driver.get('confidence_score', 0):.0f}%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    # Standard stats
    st.sidebar.markdown(f"""
    <div style="background: linear-gradient(135deg, rgba(44, 62, 80, 0.5) 0%, rgba(26, 37, 47, 0.5) 100%); 
         padding: 1rem; border-radius: 12px; margin: 1rem 0; border: 2px solid rgba(0, 191, 255, 0.3);">
        <div style="margin-bottom: 1.5rem;">
            <div style="color: #FFFFFF; font-size: 0.9rem; font-weight: 600; margin-bottom: 0.5rem;">Predicted Lap</div>
            <div style="color: #FFD700; font-size: 1.8rem; font-weight: 900;">{selected_driver['predicted_next_lap']:.3f}s</div>
        </div>
        <div style="margin-bottom: 1.5rem;">
            <div style="color: #FFFFFF; font-size: 0.9rem; font-weight: 600; margin-bottom: 0.5rem;">Best Lap</div>
            <div style="color: #FFD700; font-size: 1.6rem; font-weight: 900;">{selected_driver['best_lap']:.3f}s</div>
        </div>
        <div>
            <div style="color: #FFFFFF; font-size: 0.9rem; font-weight: 600; margin-bottom: 0.5rem;">Confidence</div>
            <div style="color: #FFD700; font-size: 1.6rem; font-weight: 900;">{selected_driver['confidence']}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

if 'cluster_label' in clusters.columns:
    driver_cluster = clusters[clusters['driver_name'] == selected_driver_name]
    if len(driver_cluster) > 0:
        st.sidebar.markdown('<hr class="divider-subtle">', unsafe_allow_html=True)
        st.sidebar.markdown("### Performance Group")
        st.sidebar.info(driver_cluster['cluster_label'].iloc[0])

# Initialize session state for active tab
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = 0

# Read from URL parameters
query_params = st.query_params
if 'tab' in query_params:
    try:
        tab_from_url = int(query_params['tab'])
        if 0 <= tab_from_url <= 4:
            st.session_state.active_tab = tab_from_url
    except ValueError:
        pass

# JavaScript to handle tab persistence - Hierarchical approach (no conflicts)
tab_persistence_js = f"""
<script>
(function() {{
    const desiredIndex = {st.session_state.active_tab};
    let isComplete = false;
    let observers = [];
    let timers = [];
    
    // Cleanup function - stops ALL strategies once we succeed
    function cleanup() {{
        if (isComplete) return;
        isComplete = true;
        
        // Disconnect all observers
        observers.forEach(obs => {{
            try {{ obs.disconnect(); }} catch(e) {{}}
        }});
        observers = [];
        
        // Clear all timers
        timers.forEach(timer => {{
            try {{ clearTimeout(timer); clearInterval(timer); }} catch(e) {{}}
        }});
        timers = [];
    }}
    
    // Get tab buttons
    function getTabButtons() {{
        let tabs = document.querySelectorAll('[data-baseweb="tab"]');
        if (!tabs || tabs.length === 0) {{
            tabs = document.querySelectorAll('[role="tab"]');
        }}
        return Array.from(tabs);
    }}
    
    // Update URL without reload
    function updateURL(tabIndex) {{
        try {{
            const url = new URL(window.parent.location.href || window.location.href);
            url.searchParams.set('tab', tabIndex);
            (window.parent.history || window.history).replaceState(null, '', url);
        }} catch(e) {{}}
    }}
    
    // Main function - restore tab and attach listeners
    function initializeTabs() {{
        if (isComplete) return true;
        
        const tabs = getTabButtons();
        if (!tabs || tabs.length === 0) return false;
        
        // Attach click listeners FIRST (before any tab switching)
        tabs.forEach((tab, idx) => {{
            if (!tab._tabListenerAttached) {{
                tab.addEventListener('click', function() {{
                    updateURL(idx);
                }}, {{ passive: true }});
                tab._tabListenerAttached = true;
            }}
        }});
        
        // Check current selection
        const currentSelected = tabs.findIndex(t => t.getAttribute('aria-selected') === 'true');
        
        // Restore desired tab if needed
        if (desiredIndex >= 0 && desiredIndex < tabs.length && currentSelected !== desiredIndex) {{
            tabs[desiredIndex].click();
        }}
        
        // Success! Stop all other strategies
        cleanup();
        return true;
    }}
    
    // HIERARCHICAL STRATEGY EXECUTION
    // Each strategy only starts if previous ones haven't succeeded
    
    // Priority 1: Immediate check (tabs might already be rendered)
    if (!initializeTabs()) {{
        
        // Priority 2: MutationObserver on tab-list (most targeted)
        const tabList = document.querySelector('[data-baseweb="tab-list"]') || 
                       document.querySelector('[role="tablist"]');
        if (tabList && !isComplete) {{
            const listObserver = new MutationObserver(() => {{
                if (!isComplete) initializeTabs();
            }});
            observers.push(listObserver);
            listObserver.observe(tabList, {{ 
                attributes: true, 
                childList: true, 
                subtree: true 
            }});
        }}
        
        // Priority 3: MutationObserver on body (broader, if tab-list not found)
        if (!isComplete) {{
            const bodyObserver = new MutationObserver(() => {{
                if (!isComplete) initializeTabs();
            }});
            observers.push(bodyObserver);
            bodyObserver.observe(document.body, {{ 
                childList: true, 
                subtree: true 
            }});
        }}
        
        // Priority 4: Timed attempts at strategic intervals
        [50, 100, 200, 300, 500, 750, 1000].forEach(delay => {{
            const timer = setTimeout(() => {{
                if (!isComplete) initializeTabs();
            }}, delay);
            timers.push(timer);
        }});
        
        // Priority 5: Polling fallback (last resort)
        let attempts = 0;
        const pollInterval = setInterval(() => {{
            attempts++;
            if (isComplete || attempts >= 40) {{  // 2 seconds max
                clearInterval(pollInterval);
            }} else {{
                initializeTabs();
            }}
        }}, 50);
        timers.push(pollInterval);
    }}
    
    // Safety cleanup after 3 seconds
    timers.push(setTimeout(cleanup, 3000));
}})();
</script>
"""

# Render the JavaScript component
st.components.v1.html(tab_persistence_js, height=0)

# Create the tabs (5 tabs - Enhanced features merged into Sector Analysis)
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Driver Overview", "Lap Time Prediction", "Driving Insights", "Sector Analysis", "Leaderboard"])

# Prepare common parameters for all tabs
tab_params = {
    'selected_driver': selected_driver,
    'selected_driver_name': selected_driver_name,
    'enhanced_mode': enhanced_mode,
    'df': None,  # Not used by pages currently
    'COLORS': COLORS,
    'apply_standard_layout': apply_standard_layout,
    'RESPONSIVE_CONFIG': RESPONSIVE_CONFIG,
    'predictions': predictions,
    'commentary': commentary,
    'clusters': clusters,
    'model_selection': model_selection,
    'driver_commentary': driver_commentary,
    'driver_clusters': driver_clusters
}

# TAB 1: Driver Overview (simplified name)
with tab1:
    overview.render(**tab_params)

# TAB 2: Lap Time Prediction (cleaner name)
with tab2:
    predictions_analysis.render(**tab_params)

# TAB 3: Driving Insights (more intuitive)
with tab3:
    insights_tab.render(**tab_params)

# TAB 4: Sector Analysis (COMPLETE - includes 6 timing points, wind, top speed)
with tab4:
    sectors.render(**tab_params)

# TAB 5: Leaderboard
with tab5:
    leaderboard.render(**tab_params)

