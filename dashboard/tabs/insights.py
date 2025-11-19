"""Driving Insights Tab - Intelligent racing analysis and actionable recommendations.

DESIGN PHILOSOPHY:
- High-contrast dark theme for professional racing analytics
- Functional color coding: Red (HIGH), Cyan (MEDIUM), Amber (WARNING), Green (SUCCESS)
- 6-section logical flow: Context → Summary → Profile → Patterns → Actions → Goals
- Standard card template: 16px radius, 6px colored left border, consistent shadows
- Strict text cleaning to prevent markdown/HTML artifacts
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import re
import os


def clean_commentary_text(text):
    """
    Strict text cleaning to remove all markdown and HTML formatting.
    Ensures professional, clean appearance with proper grammar flow.
    """
    if pd.isna(text):
        return ""
    
    text = str(text)
    
    # Handle HTML tags with proper spacing
    # Replace <strong> tags with emphasis (keep the text, remove tags)
    text = text.replace('<strong>', '**').replace('</strong>', '**')
    
    # Replace <br/> and <br> with proper sentence breaks
    text = text.replace('<br/><br/>', '. ').replace('<br/>', '. ').replace('<br>', '. ')
    
    # Remove other HTML tags
    text = text.replace('<b>', '').replace('</b>', '')
    text = text.replace('<em>', '').replace('</em>', '')
    text = text.replace('<i>', '').replace('</i>', '')
    
    # Now handle markdown bold - convert to natural emphasis
    # Pattern: **Text** becomes "Text" (keep text, remove asterisks)
    import re
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    
    # Remove em/en dashes and replace with proper punctuation
    text = text.replace('—', ' - ').replace('–', ' - ')
    
    # Clean up multiple spaces and periods
    text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single
    text = re.sub(r'\.+\s*\.+', '.', text)  # Multiple periods to single
    text = re.sub(r'\.\s+\.', '.', text)  # ". ." to "."
    
    # Ensure proper sentence spacing
    text = re.sub(r'\.\s*([A-Z])', r'. \1', text)  # Add space after period before capital
    
    # Clean up any trailing/leading spaces
    text = text.strip()
    
    # Remove trailing period-space combinations
    text = re.sub(r'\.\s*$', '.', text)
    
    return text


def render(selected_driver, selected_driver_name, enhanced_mode, df, COLORS, apply_standard_layout, 
           RESPONSIVE_CONFIG, predictions, commentary, clusters, model_selection, driver_commentary, driver_clusters):
    """
    Render driving insights tab with intelligent commentary.
    
    STRUCTURE (6 Sections):
    - FILTERS: Context setting (category, priority)
    - SECTION 1: Top Metrics (4-column summary)
    - SECTION 2: Driver Profile & Analysis (narrative context)
    - SECTION 3: Performance Patterns (technical validation)
    - SECTION 4: Actionable Recommendations (core value - prioritized actions)
    - SECTION 5: Telemetry Metrics (raw data context - conditional)
    - SECTION 6: Improvement Potential (goal visualization)
    """
    
    # ========================================================================
    # DATA LOADING - Enhanced → Original → Error
    # ========================================================================
    # Helper function to find correct path (works for both local and Streamlit Cloud)
    def get_data_path(relative_path):
        # Try from project root first (Streamlit Cloud)
        if os.path.exists(relative_path):
            return relative_path
        # Try from dashboard folder (local)
        alt_path = os.path.join('..', relative_path)
        if os.path.exists(alt_path):
            return alt_path
        return relative_path  # Return original if neither exists
    
    try:
        insights = pd.read_csv(get_data_path('outputs/enhanced_telemetry_insights/comprehensive_insights.csv'))
        insights_source = "Enhanced"
    except Exception as e:
        try:
            insights = pd.read_csv(get_data_path('outputs/telemetry_insights/actionable_insights.csv'))
            insights_source = "Original"
        except:
            # ERROR STATE: Red border for maximum visibility
            st.markdown("""
            <div style="background: rgba(44, 62, 80, 0.95); 
                 border-left: 6px solid #DC3545; 
                 padding: 1rem 1.25rem; 
                 margin: 1rem 0; 
                 border-radius: 16px; 
                 border-top: 1px solid rgba(220, 53, 69, 0.3);
                 border-right: 1px solid rgba(220, 53, 69, 0.3);
                 border-bottom: 1px solid rgba(220, 53, 69, 0.3);
                 box-shadow: 0 4px 16px rgba(0, 0, 0, 0.5), 0 2px 8px rgba(220, 53, 69, 0.2);">
                <div style="display: flex; align-items: center; gap: 1rem;">
                    <span style="font-size: 2rem;">❌</span>
                    <div>
                        <strong style="color: #FFFFFF; font-size: 1.15rem; display: block; margin-bottom: 0.25rem;">Insights Not Available</strong>
                        <p style="color: #E8E8E8; margin: 0; font-size: 0.95rem; line-height: 1.6;">
                            Run: <code style="background: rgba(0, 0, 0, 0.3); padding: 0.2rem 0.5rem; border-radius: 4px; color: #00BFFF;">py scripts/analyze_telemetry.py</code>
                        </p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            return
    
    # Get driver name - handle both old format (GR86 #88) and new format (GR86-049-88)
    driver_name_in_insights = selected_driver_name
    driver_insights = insights[insights['driver_name'] == driver_name_in_insights]
    
    # Debug: Check and clean column names
    insights.columns = insights.columns.str.strip()  # Remove any whitespace
    if 'potential_gain' not in insights.columns:
        # Find similar column names
        potential_cols = [col for col in insights.columns if 'gain' in col.lower() or 'potential' in col.lower()]
        if potential_cols:
            st.info(f"Found similar columns: {potential_cols}. Using first match.")
            # Rename to standard name
            insights = insights.rename(columns={potential_cols[0]: 'potential_gain'})
        else:
            st.warning(f"⚠️ Column 'potential_gain' not found. Available: {insights.columns.tolist()[:10]}")
    
    if len(driver_insights) == 0:
        st.markdown(f"""
        <div style="background: rgba(44, 62, 80, 0.95); 
             border-left: 6px solid #FFC107; 
             padding: 1rem 1.25rem; 
             margin: 1rem 0; 
             border-radius: 16px; 
             border: 1px solid rgba(255, 193, 7, 0.3);
             box-shadow: 0 4px 16px rgba(0, 0, 0, 0.5);">
            <strong style="color: #FFFFFF; font-size: 1.1rem;">⚠️ No insights available for {selected_driver_name}</strong>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # ========================================================================
    # CALCULATE METRICS FIRST - Before filters (so total is always accurate)
    # ========================================================================
    # Store unfiltered data for total calculations
    driver_insights_unfiltered = driver_insights.copy()
    
    high_priority = len(driver_insights_unfiltered[driver_insights_unfiltered['priority'].str.upper() == 'HIGH'])
    medium_priority = len(driver_insights_unfiltered[driver_insights_unfiltered['priority'].str.upper() == 'MEDIUM'])
    low_priority = len(driver_insights_unfiltered[driver_insights_unfiltered['priority'].str.upper() == 'LOW'])
    
    # ========================================================================
    # FILTERS - Context Setting (Enhanced mode only)
    # ========================================================================
    if insights_source == "Enhanced":
        st.markdown("### Filter Insights")
        col1, col2 = st.columns(2)
        
        with col1:
            categories = ['All Categories'] + sorted(driver_insights['category'].unique().tolist())
            selected_category = st.selectbox("Category", categories, key="category_filter")
        
        with col2:
            priorities = ['All Priorities', 'HIGH', 'MEDIUM', 'LOW']
            selected_priority = st.selectbox("Priority", priorities, key="priority_filter")
        
        # Apply filters
        if selected_category != 'All Categories':
            driver_insights = driver_insights[driver_insights['category'] == selected_category]
        
        if selected_priority != 'All Priorities':
            driver_insights = driver_insights[driver_insights['priority'].str.upper() == selected_priority]
        
        # Show skill level if available
        if 'skill_level' in driver_insights.columns and len(driver_insights) > 0:
            skill_level = driver_insights.iloc[0]['skill_level']
            st.markdown(f"""
            <div style="background: rgba(44, 62, 80, 0.5); 
                 border-left: 3px solid rgba(0, 191, 255, 0.5); 
                 padding: 0.75rem 1rem; 
                 margin: 0.25rem 0; 
                 border-radius: 8px; 
                 border: 1px solid rgba(0, 191, 255, 0.2);">
                <strong style="color: #FFFFFF;">Driver Skill Level:</strong> 
                <span style="color: #E8E8E8; font-weight: 600;">{skill_level}</span>
                <span style="color: #B0B0B0; font-size: 0.9rem;"> (based on gap to fastest lap)</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('<div style="margin: 0.25rem 0;"></div>', unsafe_allow_html=True)
    
    # ========================================================================
    # SECTION 1: TOP METRICS - Summary & Urgency (Using st.metric like Driver Overview)
    # ========================================================================
    st.markdown("### Performance Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "High Priority",
            f"{high_priority}",
            help="Critical recommendations requiring immediate attention"
        )
    
    with col2:
        st.metric(
            "Medium Priority",
            f"{medium_priority}",
            help="Important improvements for consistent gains"
        )
    
    with col3:
        st.metric(
            "Low Priority",
            f"{low_priority}",
            help="Fine-tuning opportunities for marginal gains"
        )
    
    # ========================================================================
    # SECTION
    # ========================================================================
    with st.expander("Driver Profile & Analysis", expanded=False):
        # Load intelligent commentary
        try:
            intelligent_commentary = pd.read_csv(get_data_path('outputs/commentary/ai_commentary.csv'))
            driver_commentary = intelligent_commentary[intelligent_commentary['driver_name'] == selected_driver_name]
            
            if len(driver_commentary) > 0:
                # Collect all commentary sections
                style_data = driver_commentary[driver_commentary['category'] == 'Driving Style Profile']
                expert_data = driver_commentary[driver_commentary['category'] == 'Expert Summary']
                comparative_data = driver_commentary[driver_commentary['category'] == 'Comparative Performance']
            
            # Create 2-column layout for better space usage
            col1, col2 = st.columns(2)
            
            # LEFT COLUMN: Driving Style Profile
            with col1:
                if len(style_data) > 0:
                    commentary_text = clean_commentary_text(style_data.iloc[0]['commentary'])
                    
                    # Split into style name and description
                    parts = commentary_text.split('.', 1)
                    if len(parts) == 2:
                        style_name = parts[0].strip()
                        style_description = parts[1].strip()
                    else:
                        style_name = commentary_text
                        style_description = ""
                    
                    st.markdown(f"""
                    <div style="background: rgba(44, 62, 80, 0.95); 
                         padding: 1rem 1.25rem; 
                         margin: 0.25rem 0; 
                         border-radius: 16px; 
                         border-left: 4px solid rgba(0, 191, 255, 0.6);
                         border-top: 1px solid rgba(255, 255, 255, 0.1);
                         border-right: 1px solid rgba(255, 255, 255, 0.1);
                         border-bottom: 1px solid rgba(255, 255, 255, 0.1);
                         box-shadow: 0 4px 16px rgba(0, 0, 0, 0.5);
                         height: 250px;
                         display: flex;
                         flex-direction: column;">
                        <div style="font-size: 0.85rem; 
                             font-weight: 600; 
                             color: #B0B0B0; 
                             margin-bottom: 0.25rem; 
                             letter-spacing: 1px;
                             text-transform: uppercase;">
                            Driving Style
                        </div>
                        <div style="font-size: 1.05rem; 
                             font-weight: 700; 
                             color: #FFFFFF; 
                             margin-bottom: 0.25rem;">
                            {style_name}
                        </div>
                        {f'<div style="font-size: 0.9rem; line-height: 1.6; color: #E8E8E8; flex: 1; overflow-y: auto;">{style_description}</div>' if style_description else ''}
                    </div>
                    """, unsafe_allow_html=True)
            
            # RIGHT COLUMN: Expert Summary
            with col2:
                if len(expert_data) > 0:
                    commentary_text = clean_commentary_text(expert_data.iloc[0]['commentary'])
                    st.markdown(f"""
                    <div style="background: rgba(44, 62, 80, 0.95); 
                         padding: 1rem 1.25rem; 
                         margin: 0.25rem 0; 
                         border-radius: 16px; 
                         border-left: 4px solid rgba(0, 191, 255, 0.6);
                         border-top: 1px solid rgba(255, 255, 255, 0.1);
                         border-right: 1px solid rgba(255, 255, 255, 0.1);
                         border-bottom: 1px solid rgba(255, 255, 255, 0.1);
                         box-shadow: 0 4px 16px rgba(0, 0, 0, 0.5);
                         height: 250px;
                         display: flex;
                         flex-direction: column;">
                        <div style="font-size: 0.85rem; 
                             font-weight: 600; 
                             color: #B0B0B0; 
                             margin-bottom: 0.25rem; 
                             letter-spacing: 1px;
                             text-transform: uppercase;">
                            Expert Summary
                        </div>
                        <div style="font-size: 0.9rem; line-height: 1.6; color: #E8E8E8; flex: 1; overflow-y: auto;">
                            {commentary_text}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # FULL WIDTH: Comparative Performance (if exists)
            if len(comparative_data) > 0:
                for _, comp in comparative_data.iterrows():
                    commentary_text = clean_commentary_text(comp['commentary'])
                    st.markdown(f"""
                    <div style="background: rgba(44, 62, 80, 0.95); 
                         padding: 1rem 1.25rem; 
                         margin: 0.25rem 0; 
                         border-radius: 16px; 
                         border-left: 4px solid rgba(0, 191, 255, 0.6);
                         border-top: 1px solid rgba(255, 255, 255, 0.1);
                         border-right: 1px solid rgba(255, 255, 255, 0.1);
                         border-bottom: 1px solid rgba(255, 255, 255, 0.1);
                         box-shadow: 0 4px 16px rgba(0, 0, 0, 0.5);">
                        <div style="font-size: 0.85rem; 
                             font-weight: 600; 
                             color: #B0B0B0; 
                             margin-bottom: 0.25rem; 
                             letter-spacing: 1px;
                             text-transform: uppercase;">
                            Comparative Performance
                        </div>
                        <div style="font-size: 0.9rem; line-height: 1.6; color: #E8E8E8;">
                            {commentary_text}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        except:
            st.markdown("""
            <div style="background: rgba(0, 191, 255, 0.1); 
                 border-left: 4px solid #00BFFF; 
                 padding: 0.75rem 1rem; 
                 margin: 0.25rem 0; 
                 border-radius: 8px; 
                 border: 1px solid rgba(0, 191, 255, 0.3);">
                <span style="color: #E8E8E8;">💡 Intelligent commentary not available. Run commentary generation to create AI-powered driver analysis.</span>
            </div>
            """, unsafe_allow_html=True)
    
    # ========================================================================
    # SECTION
    # ========================================================================
    with st.expander("Performance Patterns", expanded=False):
        try:
            consistency_data = driver_commentary[driver_commentary['category'] == 'Consistency Pattern Analysis']
            prediction_data = driver_commentary[driver_commentary['category'] == 'Prediction Stability Analysis']
            
            # 2-column layout for patterns
            col1, col2 = st.columns(2)
            
            # LEFT: Consistency Analysis
            with col1:
                if len(consistency_data) > 0:
                    for _, cons in consistency_data.iterrows():
                        commentary_text = clean_commentary_text(cons['commentary'])
                        st.markdown(f"""
                        <div style="background: rgba(44, 62, 80, 0.95); 
                             padding: 1rem 1.25rem; 
                             margin: 0.25rem 0; 
                             border-radius: 16px; 
                             border-left: 4px solid rgba(0, 191, 255, 0.6);
                             border-top: 1px solid rgba(255, 255, 255, 0.1);
                             border-right: 1px solid rgba(255, 255, 255, 0.1);
                             border-bottom: 1px solid rgba(255, 255, 255, 0.1);
                             box-shadow: 0 4px 16px rgba(0, 0, 0, 0.5);
                             min-height: 180px;">
                            <div style="font-size: 0.85rem; 
                                 font-weight: 600; 
                                 color: #B0B0B0; 
                                 margin-bottom: 0.25rem; 
                                 letter-spacing: 1px;
                                 text-transform: uppercase;">
                                Consistency Patterns
                            </div>
                            <div style="font-size: 0.9rem; line-height: 1.6; color: #E8E8E8;">
                                {commentary_text}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            
            # RIGHT: Prediction Stability
            with col2:
                if len(prediction_data) > 0:
                    for _, pred in prediction_data.iterrows():
                        commentary_text = clean_commentary_text(pred['commentary'])
                        st.markdown(f"""
                        <div style="background: rgba(44, 62, 80, 0.95); 
                             padding: 1rem 1.25rem; 
                             margin: 0.25rem 0; 
                             border-radius: 16px; 
                             border-left: 4px solid rgba(0, 191, 255, 0.6);
                             border-top: 1px solid rgba(255, 255, 255, 0.1);
                             border-right: 1px solid rgba(255, 255, 255, 0.1);
                             border-bottom: 1px solid rgba(255, 255, 255, 0.1);
                             box-shadow: 0 4px 16px rgba(0, 0, 0, 0.5);
                             min-height: 180px;">
                            <div style="font-size: 0.85rem; 
                                 font-weight: 600; 
                                 color: #B0B0B0; 
                                 margin-bottom: 0.25rem; 
                                 letter-spacing: 1px;
                                 text-transform: uppercase;">
                                Prediction Stability
                            </div>
                            <div style="font-size: 0.9rem; line-height: 1.6; color: #E8E8E8;">
                                {commentary_text}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
        except:
            pass
    
    # ========================================================================
    # WEATHER IMPACT ANALYSIS - Environmental Context
    # ========================================================================
    try:
        weather_data = driver_commentary[driver_commentary['category'] == 'Weather Impact Analysis']
        if len(weather_data) > 0:
            with st.expander("Weather Impact Analysis", expanded=False):
                for _, weather in weather_data.iterrows():
                    # Parse the weather commentary to extract sections
                    commentary_html = weather['commentary']
                    
                    # Extract sections using string parsing
                    import re
                    
                    # Extract Conditions Summary section
                    conditions_match = re.search(r'<strong>Conditions Summary</strong><br/><br/>(.*?)<br/><br/><strong>', commentary_html, re.DOTALL)
                    conditions_text = conditions_match.group(1) if conditions_match else ""
                    
                    # Extract Personalized Grip Assessment section
                    grip_match = re.search(r"<strong>(.*?)'s Personalized Grip Assessment</strong><br/><br/>(.*?)<br/><br/><strong>", commentary_html, re.DOTALL)
                    grip_title = grip_match.group(1) if grip_match else selected_driver_name
                    grip_text = grip_match.group(2) if grip_match else ""
                    
                    # Extract Management Strategy section
                    strategy_match = re.search(r'<strong>Actionable Management Strategy</strong><br/><br/>(.*?)$', commentary_html, re.DOTALL)
                    strategy_text = strategy_match.group(1) if strategy_match else ""
                    
                    # Clean HTML tags for display
                    def clean_html(text):
                        text = re.sub(r"<span style='color: #FFC107;'>(.*?)</span>", r'\1', text)
                        text = re.sub(r"<span style='color: #FF4444;'>(.*?)</span>", r'\1', text)
                        text = re.sub(r"<span style='color: #00FF88;'>(.*?)</span>", r'\1', text)
                        text = text.replace('<br/>', ' ')
                        return text.strip()
                    
                    conditions_clean = clean_html(conditions_text)
                    grip_clean = clean_html(grip_text)
                    strategy_clean = clean_html(strategy_text)
                    
                    # Create 2-column layout for Conditions and Grip Assessment
                    col1, col2 = st.columns(2)
                    
                    # LEFT COLUMN: Conditions Summary
                    with col1:
                        st.markdown(f"""
                        <div style="background: rgba(44, 62, 80, 0.95); 
                             padding: 1rem 1.25rem; 
                             margin: 0.25rem 0; 
                             border-radius: 16px; 
                             border-left: 4px solid rgba(0, 191, 255, 0.6);
                             border-top: 1px solid rgba(255, 255, 255, 0.1);
                             border-right: 1px solid rgba(255, 255, 255, 0.1);
                             border-bottom: 1px solid rgba(255, 255, 255, 0.1);
                             box-shadow: 0 4px 16px rgba(0, 0, 0, 0.5);
                             height: 200px;
                             display: flex;
                             flex-direction: column;">
                            <div style="font-size: 0.85rem; 
                                 font-weight: 600; 
                                 color: #B0B0B0; 
                                 margin-bottom: 0.25rem; 
                                 letter-spacing: 1px;
                                 text-transform: uppercase;">
                                Conditions Summary
                            </div>
                            <div style="font-size: 0.9rem; line-height: 1.6; color: #E8E8E8; flex: 1; overflow-y: auto;">
                                {conditions_clean}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # RIGHT COLUMN: Personalized Grip Assessment
                    with col2:
                        st.markdown(f"""
                        <div style="background: rgba(44, 62, 80, 0.95); 
                             padding: 1rem 1.25rem; 
                             margin: 0.25rem 0; 
                             border-radius: 16px; 
                             border-left: 4px solid rgba(0, 191, 255, 0.6);
                             border-top: 1px solid rgba(255, 255, 255, 0.1);
                             border-right: 1px solid rgba(255, 255, 255, 0.1);
                             border-bottom: 1px solid rgba(255, 255, 255, 0.1);
                             box-shadow: 0 4px 16px rgba(0, 0, 0, 0.5);
                             height: 200px;
                             display: flex;
                             flex-direction: column;">
                            <div style="font-size: 0.85rem; 
                                 font-weight: 600; 
                                 color: #B0B0B0; 
                                 margin-bottom: 0.25rem; 
                                 letter-spacing: 1px;
                                 text-transform: uppercase;">
                                Personalized Grip Assessment
                            </div>
                            <div style="font-size: 0.9rem; line-height: 1.6; color: #E8E8E8; flex: 1; overflow-y: auto;">
                                {grip_clean}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # FULL WIDTH: Management Strategy
                    st.markdown(f"""
                    <div style="background: rgba(44, 62, 80, 0.95); 
                         padding: 1rem 1.25rem; 
                         margin: 0.25rem 0; 
                         border-radius: 16px; 
                         border-left: 4px solid rgba(0, 191, 255, 0.6);
                         border-top: 1px solid rgba(255, 255, 255, 0.1);
                         border-right: 1px solid rgba(255, 255, 255, 0.1);
                         border-bottom: 1px solid rgba(255, 255, 255, 0.1);
                         box-shadow: 0 4px 16px rgba(0, 0, 0, 0.5);">
                        <div style="font-size: 0.85rem; 
                             font-weight: 600; 
                             color: #B0B0B0; 
                             margin-bottom: 0.25rem; 
                             letter-spacing: 1px;
                             text-transform: uppercase;">
                            Actionable Management Strategy
                        </div>
                        <div style="font-size: 0.9rem; line-height: 1.6; color: #E8E8E8;">
                            {strategy_clean}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    except:
        pass
    
    st.markdown('<div style="margin: 0.25rem 0;"></div>', unsafe_allow_html=True)
    
    # ========================================================================
    # SECTION 4: ACTIONABLE RECOMMENDATIONS - Core Value & Action
    # This is the MOST CRITICAL section - highest visual hierarchy
    # ========================================================================
    st.markdown("### Actionable Recommendations")
    
    # HIGH PRIORITY (Red border - Maximum visual dominance)
    high_priority_insights = driver_insights[driver_insights['priority'].str.upper() == 'HIGH']
    if len(high_priority_insights) > 0:
        for idx, insight in high_priority_insights.iterrows():
            # Build badges with background (matching medium priority style)
            badges = []
            if 'confidence' in insight and pd.notna(insight['confidence']):
                badges.append(f'<span style="background: rgba(0, 191, 255, 0.2); color: #FFFFFF; padding: 0.25rem 0.6rem; border-radius: 4px; font-size: 0.8rem; font-weight: 600;">Confidence: {insight["confidence"]}</span>')
            
            if 'impact_score' in insight and pd.notna(insight['impact_score']):
                try:
                    impact = float(insight['impact_score'])
                    badges.append(f'<span style="background: rgba(0, 191, 255, 0.2); color: #FFFFFF; padding: 0.25rem 0.6rem; border-radius: 4px; font-size: 0.8rem; font-weight: 600;">Impact: {impact:.3f}s</span>')
                except:
                    pass
            
            badge_html = ' '.join(badges) if badges else ''
            
            # Clean text to prevent markdown/HTML artifacts
            category = clean_commentary_text(str(insight['category']))
            issue = clean_commentary_text(str(insight['issue']))
            action = clean_commentary_text(str(insight['action']))
            
            st.markdown(f"""
            <div style="background: rgba(44, 62, 80, 0.95); 
                 padding: 1rem 1.25rem; 
                 margin: 0.25rem 0; 
                 border-radius: 16px; 
                 border-left: 5px solid rgba(220, 53, 69, 0.4);
                 border-top: 1px solid rgba(255, 255, 255, 0.1);
                 border-right: 1px solid rgba(255, 255, 255, 0.1);
                 border-bottom: 1px solid rgba(255, 255, 255, 0.1);
                 box-shadow: 0 4px 16px rgba(0, 0, 0, 0.5);
                 transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);">
                <div style="margin-bottom: 0.25rem;">
                    <strong style="font-size: 1.1rem; 
                         color: #FFFFFF; 
                         display: block; 
                         font-weight: 700;">
                        {category}
                    </strong>
                    <div style="font-size: 0.85rem; 
                         color: #E8E8E8; 
                         margin-top: 0.25rem; 
                         font-weight: 600; 
                         letter-spacing: 1px;
                         text-transform: uppercase;">
                        HIGH PRIORITY — {insight['potential_gain']}
                    </div>
                    {f'<div style="margin-top: 0.25rem; display: flex; gap: 0.5rem; flex-wrap: wrap;">{badge_html}</div>' if badge_html else ''}
                </div>
                <div style="font-size: 0.95rem; line-height: 1.8; color: #E8E8E8; margin-top: 1rem;">
                    <div style="margin-bottom: 0.25rem;">
                        <strong style="color: #FFFFFF; font-size: 0.9rem;">Issue:</strong> 
                        <span style="color: #E8E8E8;">{issue}</span>
                    </div>
                    <div>
                        <strong style="color: #FFFFFF; font-size: 0.9rem;">Action:</strong> 
                        <span style="color: #E8E8E8;">{action}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # MEDIUM & LOW PRIORITY (Side-by-side layout for better space usage)
    medium_priority_insights = driver_insights[driver_insights['priority'].str.upper() == 'MEDIUM']
    low_priority_insights = driver_insights[driver_insights['priority'].str.upper() == 'LOW']
    
    # Combine medium and low for side-by-side display
    non_high_insights = pd.concat([medium_priority_insights, low_priority_insights])
    
    if len(non_high_insights) > 0:
        # Display in 2-column grid
        for i in range(0, len(non_high_insights), 2):
            col1, col2 = st.columns(2)
            
            # First card
            insight = non_high_insights.iloc[i]
            priority_level = insight['priority'].upper()
            border_color = "rgba(0, 191, 255, 0.6)" if priority_level == 'MEDIUM' else "rgba(176, 176, 176, 0.5)"
            priority_label = "MEDIUM PRIORITY" if priority_level == 'MEDIUM' else "LOW PRIORITY"
            
            with col1:
                # Build badges - Subdued colors for medium/low priority
                badges = []
                if 'confidence' in insight and pd.notna(insight['confidence']):
                    conf_color = {
                        "High": "rgba(176, 176, 176, 0.3)",
                        "Medium": "rgba(176, 176, 176, 0.3)",
                        "Low": "rgba(176, 176, 176, 0.3)"
                    }.get(insight['confidence'], "rgba(176, 176, 176, 0.3)")
                    badges.append(f'<span style="background: {conf_color}; color: #E8E8E8; padding: 0.25rem 0.6rem; border-radius: 4px; font-size: 0.8rem; font-weight: 600;">Confidence: {insight["confidence"]}</span>')
                
                if 'impact_score' in insight and pd.notna(insight['impact_score']):
                    try:
                        impact = float(insight['impact_score'])
                        badges.append(f'<span style="background: rgba(176, 176, 176, 0.3); color: #E8E8E8; padding: 0.25rem 0.6rem; border-radius: 4px; font-size: 0.8rem; font-weight: 600;">Impact: {impact:.3f}s</span>')
                    except:
                        pass
                
                badge_html = ' '.join(badges) if badges else ''
                
                # Clean text
                category = clean_commentary_text(str(insight['category']))
                issue = clean_commentary_text(str(insight['issue']))
                action = clean_commentary_text(str(insight['action']))
                
                st.markdown(f"""
                <div style="background: rgba(44, 62, 80, 0.95); 
                     padding: 1rem 1.25rem; 
                     margin: 0.25rem 0; 
                     border-radius: 16px; 
                     border-left: 4px solid {border_color};
                     border-top: 1px solid rgba(255, 255, 255, 0.1);
                     border-right: 1px solid rgba(255, 255, 255, 0.1);
                     border-bottom: 1px solid rgba(255, 255, 255, 0.1);
                     box-shadow: 0 4px 16px rgba(0, 0, 0, 0.5);
                     min-height: 280px;
                     display: flex;
                     flex-direction: column;">
                    <div style="margin-bottom: 0.25rem;">
                        <strong style="font-size: 1rem; 
                             color: #FFFFFF; 
                             display: block; 
                             font-weight: 700;">
                            {category}
                        </strong>
                        <div style="font-size: 0.8rem; 
                             color: #B0B0B0; 
                             margin-top: 0.25rem; 
                             font-weight: 600; 
                             letter-spacing: 1px;
                             text-transform: uppercase;">
                            {priority_label} — {insight['potential_gain']}
                        </div>
                        {f'<div style="margin-top: 0.25rem; display: flex; gap: 0.5rem; flex-wrap: wrap;">{badge_html}</div>' if badge_html else ''}
                    </div>
                    <div style="font-size: 0.9rem; line-height: 1.7; color: #E8E8E8; flex: 1;">
                        <div style="margin-bottom: 0.25rem;">
                            <strong style="color: #FFFFFF; font-size: 0.85rem;">Issue:</strong> 
                            <span style="color: #E8E8E8;">{issue}</span>
                        </div>
                        <div>
                            <strong style="color: #FFFFFF; font-size: 0.85rem;">Action:</strong> 
                            <span style="color: #E8E8E8;">{action}</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Second card (if exists)
            if i + 1 < len(non_high_insights):
                insight = non_high_insights.iloc[i + 1]
                priority_level = insight['priority'].upper()
                border_color = "rgba(0, 191, 255, 0.6)" if priority_level == 'MEDIUM' else "rgba(176, 176, 176, 0.5)"
                priority_label = "MEDIUM PRIORITY" if priority_level == 'MEDIUM' else "LOW PRIORITY"
                
                with col2:
                    # Build badges - Subdued colors for medium/low priority
                    badges = []
                    if 'confidence' in insight and pd.notna(insight['confidence']):
                        conf_color = {
                            "High": "rgba(176, 176, 176, 0.3)",
                            "Medium": "rgba(176, 176, 176, 0.3)",
                            "Low": "rgba(176, 176, 176, 0.3)"
                        }.get(insight['confidence'], "rgba(176, 176, 176, 0.3)")
                        badges.append(f'<span style="background: {conf_color}; color: #E8E8E8; padding: 0.25rem 0.6rem; border-radius: 4px; font-size: 0.8rem; font-weight: 600;">Confidence: {insight["confidence"]}</span>')
                    
                    if 'impact_score' in insight and pd.notna(insight['impact_score']):
                        try:
                            impact = float(insight['impact_score'])
                            badges.append(f'<span style="background: rgba(176, 176, 176, 0.3); color: #E8E8E8; padding: 0.25rem 0.6rem; border-radius: 4px; font-size: 0.8rem; font-weight: 600;">Impact: {impact:.3f}s</span>')
                        except:
                            pass
                    
                    badge_html = ' '.join(badges) if badges else ''
                    
                    # Clean text
                    category = clean_commentary_text(str(insight['category']))
                    issue = clean_commentary_text(str(insight['issue']))
                    action = clean_commentary_text(str(insight['action']))
                    
                    st.markdown(f"""
                    <div style="background: rgba(44, 62, 80, 0.95); 
                         padding: 1rem 1.25rem; 
                         margin: 0.25rem 0; 
                         border-radius: 16px; 
                         border-left: 4px solid {border_color};
                         border-top: 1px solid rgba(255, 255, 255, 0.1);
                         border-right: 1px solid rgba(255, 255, 255, 0.1);
                         border-bottom: 1px solid rgba(255, 255, 255, 0.1);
                         box-shadow: 0 4px 16px rgba(0, 0, 0, 0.5);
                         min-height: 280px;
                         display: flex;
                         flex-direction: column;">
                        <div style="margin-bottom: 0.25rem;">
                            <strong style="font-size: 1rem; 
                                 color: #FFFFFF; 
                                 display: block; 
                                 font-weight: 700;">
                                {category}
                            </strong>
                            <div style="font-size: 0.8rem; 
                                 color: #B0B0B0; 
                                 margin-top: 0.25rem; 
                                 font-weight: 600; 
                                 letter-spacing: 1px;
                                 text-transform: uppercase;">
                                {priority_label} — {insight['potential_gain']}
                            </div>
                            {f'<div style="margin-top: 0.25rem; display: flex; gap: 0.5rem; flex-wrap: wrap;">{badge_html}</div>' if badge_html else ''}
                        </div>
                        <div style="font-size: 0.9rem; line-height: 1.7; color: #E8E8E8; flex: 1;">
                            <div style="margin-bottom: 0.25rem;">
                                <strong style="color: #FFFFFF; font-size: 0.85rem;">Issue:</strong> 
                                <span style="color: #E8E8E8;">{issue}</span>
                            </div>
                            <div>
                                <strong style="color: #FFFFFF; font-size: 0.85rem;">Action:</strong> 
                                <span style="color: #E8E8E8;">{action}</span>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    
    st.markdown('<div style="margin: 0.25rem 0;"></div>', unsafe_allow_html=True)
    
    # ========================================================================
    # SECTION 5: TELEMETRY METRICS - Raw Data Context (Conditional)
    # Only displayed for original insights (legacy support)
    # ========================================================================
    if 'avg_speed' in driver_insights.columns:
        st.markdown("### Telemetry Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div style="background: rgba(44, 62, 80, 0.95); 
                 padding: 1rem 1.25rem; 
                 margin: 0.25rem 0; 
                 border-radius: 16px; 
                 border-left: 6px solid #00BFFF;
                 border: 1px solid rgba(0, 191, 255, 0.2);
                 box-shadow: 0 4px 16px rgba(0, 0, 0, 0.5);
                 min-height: 160px;">
                <div style="font-size: 0.85rem; 
                     font-weight: 600; 
                     color: #B0B0B0; 
                     margin-bottom: 1rem; 
                     letter-spacing: 1px;
                     text-transform: uppercase;">
                    Speed & Throttle
                </div>
                <div style="font-size: 0.95rem; line-height: 2; color: #E8E8E8;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                        <strong style="color: #FFFFFF;">Average Speed:</strong> 
                        <span style="color: #00BFFF; font-weight: 700;">{driver_insights.iloc[0]['avg_speed']:.1f} km/h</span>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <strong style="color: #FFFFFF;">Average Throttle:</strong> 
                        <span style="color: #00BFFF; font-weight: 700;">{driver_insights.iloc[0]['avg_throttle']:.1f}%</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="background: rgba(44, 62, 80, 0.95); 
                 padding: 1rem 1.25rem; 
                 margin: 0.25rem 0; 
                 border-radius: 16px; 
                 border-left: 6px solid #00BFFF;
                 border: 1px solid rgba(0, 191, 255, 0.2);
                 box-shadow: 0 4px 16px rgba(0, 0, 0, 0.5);
                 min-height: 160px;">
                <div style="font-size: 0.85rem; 
                     font-weight: 600; 
                     color: #B0B0B0; 
                     margin-bottom: 1rem; 
                     letter-spacing: 1px;
                     text-transform: uppercase;">
                    Braking & RPM
                </div>
                <div style="font-size: 0.95rem; line-height: 2; color: #E8E8E8;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                        <strong style="color: #FFFFFF;">Average Brake:</strong> 
                        <span style="color: #00BFFF; font-weight: 700;">{driver_insights.iloc[0]['avg_brake']:.1f}%</span>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <strong style="color: #FFFFFF;">Average RPM:</strong> 
                        <span style="color: #00BFFF; font-weight: 700;">{driver_insights.iloc[0]['avg_rpm']:.0f}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Standard spacing before final section
        st.markdown('<div style="margin: 0.25rem 0;"></div>', unsafe_allow_html=True)
    
    # ========================================================================
    # SECTION 6: IMPROVEMENT POTENTIAL - Goal Setting & Visualization
    # ========================================================================
    st.markdown("### Improvement Potential")
    
    # Calculate improvements from high priority items only
    improvements = []
    current_best_str = driver_insights.iloc[0]['current_best_lap']
    
    # Convert current_best to float (handles "96.863s" format)
    try:
        current_best = float(str(current_best_str).replace('s', '').strip())
    except:
        current_best = 0.0
    
    # Parse gains from high priority insights
    for _, insight in high_priority_insights.iterrows():
        try:
            gain_str = str(insight['potential_gain'])
            numbers = re.findall(r'\d+\.?\d*', gain_str)
            if numbers:
                gain_val = float(numbers[0])
                category = clean_commentary_text(str(insight['category']))
                improvements.append({'category': category, 'gain': gain_val})
        except:
            continue
    
    if len(improvements) > 0 and sum(imp['gain'] for imp in improvements) > 0 and current_best > 0:
        total_potential = sum(imp['gain'] for imp in improvements)
        projected_best = current_best - total_potential
        
        # Build waterfall chart data
        categories = ['Current\nBest'] + [imp['category'] for imp in improvements] + ['Projected\nBest']
        values = [current_best] + [-imp['gain'] for imp in improvements] + [0]
        
        # Create waterfall visualization
        fig = go.Figure()
        
        # Calculate percentages for each improvement relative to total
        improvement_percentages = [(imp['gain'] / total_potential) * 100 for imp in improvements]
        
        fig.add_trace(go.Waterfall(
            name="Improvement",
            orientation="v",
            measure=["absolute"] + ["relative"] * len(improvements) + ["total"],
            x=categories,
            y=values,
            text=[f"<b>{val:.3f}s</b>" if val != 0 else f"<b>{projected_best:.3f}s</b>" for val in values],
            textposition="outside",
            textfont=dict(size=12, color='#FFFFFF', family='Arial, sans-serif'),
            connector={"line": {"color": "rgba(255, 255, 255, 0.3)", "width": 2, "dash": "dot"}},
            # Green for improvements (decreasing time) - show as percentage of total
            decreasing={"marker": {"color": "#28A745", "line": {"color": "rgba(255, 255, 255, 0.2)", "width": 2}}},
            # Red for increases (shouldn't happen, but handle it)
            increasing={"marker": {"color": "#DC3545", "line": {"color": "rgba(255, 255, 255, 0.2)", "width": 2}}},
            # Cyan for totals (current and projected)
            totals={"marker": {"color": "#00BFFF", "line": {"color": "rgba(255, 255, 255, 0.2)", "width": 2}}},
            # Add percentage labels
            customdata=[[f"{pct:.1f}%"] for pct in [0] + improvement_percentages + [0]],
            hovertemplate='<b>%{x}</b><br>Time: %{y:.3f}s<br>% of Total: %{customdata[0]}<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(
                text=f"<b>Cumulative Improvement Potential: {total_potential:.3f}s Faster</b>",
                font=dict(size=20, color='#FFFFFF', family='Arial, sans-serif'),
                x=0.5, xanchor='center', y=0.95, yanchor='top'
            ),
            xaxis_title=dict(text="", font=dict(size=12, color='#FFFFFF', family='Arial, sans-serif')),
            yaxis_title=dict(
                text="<b>Lap Time (seconds)</b>", 
                font=dict(size=12, color='#FFFFFF', family='Arial, sans-serif')
            ),
            height=550, 
            showlegend=False,
            plot_bgcolor='rgba(26, 37, 47, 0.5)',
            paper_bgcolor='rgba(44, 62, 80, 0.3)',
            font=dict(color='#FFFFFF', size=14, family='Arial, sans-serif'),
            margin=dict(t=140, b=80, l=90, r=60),
            xaxis=dict(
                showgrid=False, 
                tickfont=dict(size=11, color='#FFFFFF'), 
                showline=True, 
                linewidth=2, 
                linecolor='rgba(0, 191, 255, 0.5)'
            ),
            yaxis=dict(
                showgrid=True, 
                gridcolor='rgba(255, 255, 255, 0.1)', 
                tickfont=dict(size=11, color='#FFFFFF'), 
                showline=True, 
                linewidth=2, 
                linecolor='rgba(0, 191, 255, 0.5)',
                # Zoom into relevant range - add 10% padding above and below
                range=[projected_best * 0.99, current_best * 1.01]
            )
        )
        
        # Render chart with responsive config
        chart_config = {
            'displayModeBar': True, 
            'displaylogo': False, 
            'modeBarButtonsToRemove': ['lasso2d', 'select2d'], 
            'responsive': True
        }
        st.plotly_chart(fig, use_container_width=True, config=chart_config, key="improvement_waterfall")
        
        # SUCCESS SUMMARY BOX (Cyan accent - consistent with design)
        percentage_improvement = (total_potential / current_best) * 100
        st.markdown(f"""
        <div style="background: rgba(44, 62, 80, 0.95); 
             border-left: 6px solid rgba(0, 191, 255, 0.6); 
             padding: 1rem 1.25rem; 
             margin: 0.75rem 0; 
             border-radius: 16px; 
             border-top: 1px solid rgba(255, 255, 255, 0.1);
             border-right: 1px solid rgba(255, 255, 255, 0.1);
             border-bottom: 1px solid rgba(255, 255, 255, 0.1);
             box-shadow: 0 4px 16px rgba(0, 0, 0, 0.5);">
            <div style="font-size: 1.1rem; 
                 font-weight: 700; 
                 color: #FFFFFF; 
                 margin-bottom: 1rem; 
                 letter-spacing: 0.5px;">
                TOTAL POTENTIAL IMPROVEMENT
            </div>
            <div style="font-size: 1.05rem; line-height: 1.8; color: #E8E8E8;">
                <div style="margin-bottom: 0.25rem;">
                    Implementing all <strong style="color: #FFFFFF;">HIGH PRIORITY</strong> actions could improve lap time by 
                    <strong style="color: #00BFFF; font-size: 1.2rem;">{total_potential:.3f}s</strong>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; 
                     background: rgba(0, 0, 0, 0.2); padding: 1rem; border-radius: 8px; margin-top: 1rem;">
                    <div>
                        <div style="color: #B0B0B0; font-size: 0.9rem; margin-bottom: 0.25rem;">Current Best Lap</div>
                        <div style="color: #FFFFFF; font-size: 1.3rem; font-weight: 700;">{current_best:.3f}s</div>
                    </div>
                    <div style="color: #00BFFF; font-size: 2rem;">→</div>
                    <div>
                        <div style="color: #B0B0B0; font-size: 0.9rem; margin-bottom: 0.25rem;">Projected Best Lap</div>
                        <div style="color: #00BFFF; font-size: 1.3rem; font-weight: 700;">{projected_best:.3f}s</div>
                    </div>
                    <div>
                        <div style="color: #B0B0B0; font-size: 0.9rem; margin-bottom: 0.25rem;">Improvement</div>
                        <div style="color: #FFFFFF; font-size: 1.3rem; font-weight: 700;">{percentage_improvement:.2f}%</div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # No quantifiable data available
        st.markdown("""
        <div style="background: rgba(0, 191, 255, 0.1); 
             border-left: 4px solid #00BFFF; 
             padding: 1rem 1.25rem; 
             margin: 1rem 0; 
             border-radius: 12px; 
             border: 1px solid rgba(0, 191, 255, 0.3);">
            <strong style="color: #00BFFF; font-size: 1.1rem;">📊 No Quantifiable Improvement Data</strong>
            <p style="color: #E8E8E8; margin: 0.25rem 0 0 0; line-height: 1.6;">
                Focus on implementing the actionable recommendations above. 
                Improvement potential will be calculated once high-priority insights are available.
            </p>
        </div>
        """, unsafe_allow_html=True)


