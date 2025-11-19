"""
Reusable Metric Display Components
"""

import streamlit as st


def display_driver_metrics(selected_driver, enhanced_mode):
    """Display the 5 key driver metrics in centered layout"""
    st.markdown('<div style="display: flex; justify-content: center;">', unsafe_allow_html=True)
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Best Lap",
            f"{selected_driver['best_lap']:.3f}s",
            help="Fastest lap time achieved"
        )
    
    with col2:
        st.metric(
            "Average Lap",
            f"{selected_driver['avg_lap']:.3f}s",
            help="Average lap time across all laps"
        )
    
    with col3:
        st.metric(
            "Predicted Next",
            f"{selected_driver['predicted_next_lap']:.3f}s",
            help="ML-predicted next lap time"
        )
    
    with col4:
        improvement = selected_driver['improvement_vs_best']
        arrow = "↓" if improvement < 0 else "↑"
        st.metric(
            "Gap to Best",
            f"{arrow} {improvement:.3f}s",
            help="Difference from best lap (↓ faster, ↑ slower)"
        )
    
    with col5:
        if enhanced_mode:
            confidence_score = selected_driver.get('confidence_score', 0)
            st.metric(
                "Confidence",
                f"{confidence_score:.0f}%",
                help="ML model confidence score (0-100%)"
            )
        else:
            confidence = selected_driver.get('confidence', 'N/A')
            st.metric(
                "Confidence",
                f"{confidence}",
                help="Prediction confidence level"
            )
    
    st.markdown('</div>', unsafe_allow_html=True)


def display_sector_metrics(driver_sector):
    """Display sector time metrics"""
    st.markdown('<div style="display: flex; justify-content: center;">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Sector 1",
            f"{driver_sector['s1_best']:.3f}s",
            delta=f"Avg: {driver_sector['s1_avg']:.3f}s",
            help=f"Best S1 time | Consistency: {driver_sector['s1_consistency']:.1f}%"
        )
    
    with col2:
        st.metric(
            "Sector 2",
            f"{driver_sector['s2_best']:.3f}s",
            delta=f"Avg: {driver_sector['s2_avg']:.3f}s",
            help=f"Best S2 time | Consistency: {driver_sector['s2_consistency']:.1f}%"
        )
    
    with col3:
        st.metric(
            "Sector 3",
            f"{driver_sector['s3_best']:.3f}s",
            delta=f"Avg: {driver_sector['s3_avg']:.3f}s",
            help=f"Best S3 time | Consistency: {driver_sector['s3_consistency']:.1f}%"
        )
    
    st.markdown('</div>', unsafe_allow_html=True)


def display_warning_box(title, message):
    """Display a warning box"""
    st.markdown(f"""
    <div style="background: rgba(44, 62, 80, 0.3); border-left: 6px solid #FFC107; padding: 1rem 1.5rem; 
         margin: 1rem 0; border-radius: 16px; border: 1px solid rgba(255, 193, 7, 0.3);">
        <div style="display: flex; align-items: center; gap: 1rem;">
            <span style="font-size: 2rem;">⚠️</span>
            <div>
                <strong style="color: #FFFFFF; font-size: 1.1rem;">{title}</strong>
                <p style="color: #FFFFFF; margin: 0.5rem 0 0 0; font-size: 0.95rem;">
                    {message}
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
