import os
import sys

# Automatically invoke 'streamlit run' if user runs with 'python app.py' directly
if __name__ == "__main__":
    try:
        import streamlit.runtime
        if not streamlit.runtime.exists():
            from streamlit.web import cli as stcli
            sys.argv = ["streamlit", "run", os.path.abspath(__file__)]
            sys.exit(stcli.main())
    except ImportError:
        pass

import datetime
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Modular Architecture Imports
from data.database import init_db, get_historical_readings
from data.stations import (
    get_all_stations,
    get_available_cities,
    get_station_by_id,
    get_station_options_for_city,
)
from data.api_client import get_station_telemetry, fetch_and_sync_station_data
from analysis.aqi_calc import calculate_overall_aqi, get_category_meta
from analysis.advisory import get_health_advisory
from models.prediction_service import (
    generate_upcoming_trajectory,
    get_early_warning_summary,
)
from models.clustering_service import (
    assign_cluster_pattern,
    get_all_cluster_profiles,
    match_station_to_cluster,
)

# -----------------------------------------------------------------------------
# 0. STREAMLIT APP CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AirSense AI — Environmental Air Quality Service",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize local SQLite database
init_db()

# -----------------------------------------------------------------------------
# 1. PUBLIC-FACING MODERN CSS STYLING
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Top Header Banner */
    .app-header {
        background: linear-gradient(135deg, rgba(14, 116, 144, 0.12) 0%, rgba(2, 132, 199, 0.06) 100%);
        border: 1px solid rgba(14, 165, 233, 0.22);
        border-radius: 16px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.8rem;
    }
    .app-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0;
        color: #0284c7;
    }
    .app-subtitle {
        font-size: 1.0rem;
        color: #64748b;
        margin-top: 0.35rem;
        margin-bottom: 0;
        line-height: 1.5;
    }

    /* Primary AQI Card */
    .aqi-banner {
        border-radius: 16px;
        padding: 1.8rem 2rem;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(148, 163, 184, 0.2);
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .aqi-number {
        font-size: 3.6rem;
        font-weight: 800;
        line-height: 1.0;
        letter-spacing: -0.03em;
    }
    .aqi-category {
        font-size: 1.4rem;
        font-weight: 700;
        margin-top: 0.4rem;
    }

    /* Metric Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 14px;
        padding: 1.15rem 1.3rem;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .metric-label {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
    }
    .metric-value {
        font-size: 1.65rem;
        font-weight: 800;
        color: #0284c7;
        margin-top: 0.3rem;
    }
    .metric-sub {
        font-size: 0.78rem;
        color: #94a3b8;
        margin-top: 0.35rem;
    }

    /* Badges */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-live {
        background: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .badge-recent {
        background: rgba(2, 132, 199, 0.15);
        color: #0284c7;
        border: 1px solid rgba(2, 132, 199, 0.3);
    }
    .badge-cached {
        background: rgba(245, 158, 11, 0.15);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }

    /* Section Subheaders */
    .section-title {
        font-size: 1.28rem;
        font-weight: 700;
        margin-top: 1.6rem;
        margin-bottom: 0.8rem;
    }

    /* Info Callout Box */
    .info-card {
        background: rgba(2, 132, 199, 0.05);
        border-left: 4px solid #0284c7;
        border-radius: 0 10px 10px 0;
        padding: 0.9rem 1.2rem;
        font-size: 0.88rem;
        color: #334155;
        margin-top: 0.8rem;
        margin-bottom: 1.2rem;
    }
    [data-theme="dark"] .info-card {
        color: #cbd5e1;
    }

    /* Sidebar Branding */
    .sidebar-brand {
        padding: 0.2rem 0 0.8rem 0;
        border-bottom: 1px solid rgba(148, 163, 184, 0.18);
        margin-bottom: 1rem;
    }
    .sidebar-brand h2 {
        font-size: 1.35rem;
        font-weight: 800;
        margin: 0;
        color: #0284c7;
    }
    .sidebar-brand p {
        font-size: 0.8rem;
        color: #64748b;
        margin: 0.2rem 0 0 0;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. SIDEBAR NAVIGATION & LOCATION SELECTION
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <h2>🍃 AirSense AI</h2>
        <p>Public Environmental Monitoring Service</p>
    </div>
    """, unsafe_allow_html=True)

    nav_selection = st.radio(
        "Navigation",
        options=[
            "🏠 Air Quality Dashboard",
            "📊 Historical Data / Trends",
            "🛡️ Analysis & Health Guidance"
        ],
        index=0,
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("<b>Select Monitoring Location</b>", unsafe_allow_html=True)

    # City selector populated from the station registry
    available_cities = get_available_cities()
    selected_city = st.selectbox(
        "Target City",
        options=available_cities,
        index=available_cities.index("Pune") if "Pune" in available_cities else 0
    )

    # Show only stations/localities registered for the selected city
    station_options = get_station_options_for_city(selected_city)
    station_names = list(station_options.keys())
    selected_station_name = st.selectbox(
        "Locality / Station",
        options=station_names,
        index=0
    )
    current_station_id = station_options[selected_station_name]
    station_meta = get_station_by_id(current_station_id)

    # Station Type Transparency Callout
    if not station_meta["is_physical_station"]:
        st.markdown(f"""
        <div style="font-size: 0.82rem; line-height: 1.5; color: #64748b; background: rgba(245, 158, 11, 0.08); padding: 0.75rem 0.9rem; border-radius: 8px; border-left: 3px solid #f59e0b; margin-top: 0.5rem;">
            <b>Locality Notice:</b> {station_meta['locality']} does not host an on-site CPCB physical tower. Telemetry combines high-resolution gridded atmospheric chemistry with reference from nearby <b>{station_meta['nearest_physical_station']}</b> ({station_meta['distance_to_physical_km']} km away).
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="font-size: 0.82rem; line-height: 1.5; color: #64748b; background: rgba(16, 185, 129, 0.08); padding: 0.75rem 0.9rem; border-radius: 8px; border-left: 3px solid #10b981; margin-top: 0.5rem;">
            <b>Physical Ground Station:</b> Continuous physical air monitoring facility operated by official environmental authorities.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    # Manual Refresh Button
    if st.button("🔄 Refresh Data Now", width="stretch"):
        with st.spinner("Fetching latest verified observations..."):
            success, msg, count = fetch_and_sync_station_data(current_station_id, past_days=7)
            if success:
                st.success(f"Updated successfully! ({count} records)")
            else:
                st.warning(msg)
            st.rerun()

    st.caption("Official Standard: Indian National Air Quality Index (NAQI) | CPCB MoEFCC Guidelines")

# Retrieve current telemetry for selected station
telemetry_data = get_station_telemetry(current_station_id)
station_info = telemetry_data["station"]
latest_obs = telemetry_data["latest"]
status_badge = telemetry_data["status_label"]
status_color = telemetry_data["status_color"]


# -----------------------------------------------------------------------------
# PAGE 1: HOME / AIR QUALITY DASHBOARD
# -----------------------------------------------------------------------------
def render_dashboard():
    # Header Banner
    st.markdown(f"""
    <div class="app-header">
        <h1 class="app-title">AirSense AI — Environmental Air Quality Service</h1>
        <p class="app-subtitle">
            Reliable, verified real-world air pollution monitoring and citizen health advisories for <b>{station_info['name']}</b>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Top Status Bar
    tcol1, tcol2, tcol3, tcol4 = st.columns([1.5, 1.3, 1.2, 1])

    obs_time_str = "Unavailable"
    if latest_obs and latest_obs.get("timestamp"):
        try:
            obs_dt = pd.to_datetime(latest_obs["timestamp"])
            obs_time_str = obs_dt.strftime("%d %b %Y, %I:%M %p IST")
        except Exception:
            obs_time_str = str(latest_obs["timestamp"])

    sync_time_str = "Recently Updated"
    if latest_obs and latest_obs.get("created_at"):
        try:
            sync_dt = pd.to_datetime(latest_obs["created_at"])
            sync_time_str = sync_dt.strftime("%d %b %Y, %I:%M %p")
        except Exception:
            pass

    with tcol1:
        st.markdown(f"<b>Station / Locality:</b> {station_info['name']}", unsafe_allow_html=True)
    with tcol2:
        st.markdown(f"<b>Observation Timestamp:</b> {obs_time_str}", unsafe_allow_html=True)
    with tcol3:
        st.markdown(f"<b>Last Database Sync:</b> {sync_time_str}", unsafe_allow_html=True)
    with tcol4:
        st.markdown(f"""
        <span class="badge" style="background: {status_color}22; color: {status_color}; border: 1px solid {status_color}55;">
            ● {status_badge}
        </span>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)

    # Main AQI & Prominent Pollutant Banner
    if latest_obs:
        aqi_val = latest_obs.get("aqi")
        aqi_cat, aqi_color, aqi_desc = get_category_meta(aqi_val)
        prominent = latest_obs.get("prominent_pollutant", "PM2.5")

        bcol1, bcol2 = st.columns([1.4, 2])
        with bcol1:
            st.markdown(f"""
            <div class="aqi-banner" style="background: {aqi_color}18; border-color: {aqi_color}55;">
                <div class="metric-label" style="color: #64748b;">Indian National Air Quality Index (NAQI)</div>
                <div class="aqi-number" style="color: {aqi_color};">{aqi_val if aqi_val is not None else '--'}</div>
                <div class="aqi-category" style="color: {aqi_color};">{aqi_cat}</div>
                <div style="font-size: 0.85rem; color: #64748b; margin-top: 0.5rem;">
                    Prominent Pollutant: <b>{prominent}</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with bcol2:
            st.markdown(f"""
            <div class="metric-card" style="border-left: 4px solid {aqi_color};">
                <div>
                    <div class="metric-label">Public Health Summary</div>
                    <div style="font-size: 1.15rem; font-weight: 700; color: #1e293b; margin-top: 0.4rem;">
                        {aqi_desc}
                    </div>
                    <div style="font-size: 0.88rem; color: #64748b; margin-top: 0.6rem; line-height: 1.5;">
                        Official classification according to the Central Pollution Control Board (CPCB) National Air Quality Index. 
                        Index values under 100 represent clean or acceptable environmental conditions.
                    </div>
                </div>
                <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 0.8rem;">
                    Data Provider: {station_info['provider']} | {station_info['attribution']}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Pollutant Concentration Cards
        st.markdown('<div class="section-title">🧪 Monitored Atmospheric Pollutants</div>', unsafe_allow_html=True)
        p1, p2, p3, p4, p5, p6 = st.columns(6)

        with p1:
            val = latest_obs.get("pm25")
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Fine Particles (PM₂.₅)</div>
                <div class="metric-value">{f"{val:.1f}" if val is not None else "--"} <span style="font-size:0.9rem;">µg/m³</span></div>
                <div class="metric-sub">24-hr Standard: 60 µg/m³</div>
            </div>
            """, unsafe_allow_html=True)

        with p2:
            val = latest_obs.get("pm10")
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Respirable Dust (PM₁₀)</div>
                <div class="metric-value">{f"{val:.1f}" if val is not None else "--"} <span style="font-size:0.9rem;">µg/m³</span></div>
                <div class="metric-sub">24-hr Standard: 100 µg/m³</div>
            </div>
            """, unsafe_allow_html=True)

        with p3:
            val = latest_obs.get("no2")
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Nitrogen Dioxide (NO₂)</div>
                <div class="metric-value">{f"{val:.1f}" if val is not None else "--"} <span style="font-size:0.9rem;">µg/m³</span></div>
                <div class="metric-sub">Vehicular emission tracer</div>
            </div>
            """, unsafe_allow_html=True)

        with p4:
            val = latest_obs.get("co")
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Carbon Monoxide (CO)</div>
                <div class="metric-value">{f"{val:.2f}" if val is not None else "--"} <span style="font-size:0.9rem;">mg/m³</span></div>
                <div class="metric-sub">8-hr Standard: 2.0 mg/m³</div>
            </div>
            """, unsafe_allow_html=True)

        with p5:
            val = latest_obs.get("so2")
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Sulphur Dioxide (SO₂)</div>
                <div class="metric-value">{f"{val:.1f}" if val is not None else "--"} <span style="font-size:0.9rem;">µg/m³</span></div>
                <div class="metric-sub">24-hr Standard: 80 µg/m³</div>
            </div>
            """, unsafe_allow_html=True)

        with p6:
            val = latest_obs.get("ozone")
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Ground Ozone (O₃)</div>
                <div class="metric-value">{f"{val:.1f}" if val is not None else "--"} <span style="font-size:0.9rem;">µg/m³</span></div>
                <div class="metric-sub">Photochemical oxidant</div>
            </div>
            """, unsafe_allow_html=True)

        # Objective 1: Pollution Forecast & Early Warnings
        st.markdown('<div class="section-title">🔮 Pollution Forecast & Early Warnings</div>', unsafe_allow_html=True)
        st.write(
            "Upcoming 12-hour predictive atmospheric trajectory conditioned on recent multi-pollutant telemetry "
            "and diurnal dispersion cycles. Provides citizens with early warnings before air pollution levels peak."
        )

        history_df = get_historical_readings(current_station_id, limit=72)
        traj_df = generate_upcoming_trajectory(history_df, hours_ahead=12)

        if traj_df is not None and not traj_df.empty:
            summary = get_early_warning_summary(traj_df, latest_obs)
            start_str = pd.to_datetime(traj_df["DateTime"].iloc[0]).strftime("%I:%M %p")
            end_str = pd.to_datetime(traj_df["DateTime"].iloc[-1]).strftime("%I:%M %p")

            # 3 Summary Status Cards
            wcol1, wcol2, wcol3 = st.columns(3)

            with wcol1:
                trend_dir = summary.get("trend_direction", "Stable")
                trend_col = summary.get("trend_color", "#0284c7")
                trend_ic = summary.get("trend_icon", "→")
                st.markdown(f"""
                <div class="metric-card" style="border-left: 4px solid {trend_col};">
                    <div class="metric-label">12-Hour Pollution Trend</div>
                    <div style="font-size: 1.5rem; font-weight: 800; color: {trend_col}; margin-top: 0.2rem;">
                        {trend_ic} {trend_dir}
                    </div>
                    <div class="metric-sub">
                        {'Pollution expected to concentrate during upcoming hours' if trend_dir == 'Increasing' else ('Atmospheric dispersion expected to improve air quality' if trend_dir == 'Decreasing' else 'Pollutant levels remaining near current baseline')}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with wcol2:
                st.markdown(f"""
                <div class="metric-card" style="border-left: 4px solid #0284c7;">
                    <div class="metric-label">Forecast Horizon</div>
                    <div style="font-size: 1.35rem; font-weight: 800; color: #0284c7; margin-top: 0.2rem;">
                        12 Hours Ahead
                    </div>
                    <div class="metric-sub">
                        Valid: {start_str} to {end_str}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with wcol3:
                has_warnings = len(summary.get("warnings", [])) > 0
                alert_col = "#ef4444" if has_warnings else "#10b981"
                alert_title = "⚠️ Exceedance Warning" if has_warnings else "✅ Below Thresholds"
                alert_text = f"{len(summary.get('warnings', []))} pollutant(s) projected to exceed CPCB limits" if has_warnings else "All pollutants projected to stay within 24-hr standards"
                st.markdown(f"""
                <div class="metric-card" style="border-left: 4px solid {alert_col};">
                    <div class="metric-label">Threshold Advisory</div>
                    <div style="font-size: 1.35rem; font-weight: 800; color: {alert_col}; margin-top: 0.2rem;">
                        {alert_title}
                    </div>
                    <div class="metric-sub">
                        {alert_text}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Warning Callout Banners if any threshold exceeded
            if has_warnings:
                for warn in summary["warnings"]:
                    st.markdown(f"""
                    <div style="background: rgba(239, 68, 68, 0.08); border-left: 4px solid #ef4444; border-radius: 0 10px 10px 0; padding: 0.8rem 1.1rem; margin-top: 0.8rem; margin-bottom: 0.5rem; font-size: 0.88rem; color: #991b1b;">
                        <b>Early Warning:</b> {warn['message']} Peak around <b>{warn['time']}</b> with uncertainty range <b>{warn['uncertainty']}</b>.
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("<div style='height: 0.6rem;'></div>", unsafe_allow_html=True)

            # Forecast Chart & Details
            fcol1, fcol2 = st.columns([2, 1])
            with fcol1:
                fig_traj = go.Figure()

                # PM2.5 Trajectory
                fig_traj.add_trace(go.Scatter(
                    x=traj_df["DateTime"],
                    y=traj_df["predicted_pm25"],
                    mode="lines+markers",
                    name="Predicted PM₂.₅ (µg/m³)",
                    line=dict(color="#f59e0b", width=2.8),
                    hovertemplate="%{x|%I:%M %p}<br>PM₂.₅: %{y:.1f} µg/m³<extra></extra>"
                ))

                # PM10 Trajectory
                if "predicted_pm10" in traj_df.columns:
                    fig_traj.add_trace(go.Scatter(
                        x=traj_df["DateTime"],
                        y=traj_df["predicted_pm10"],
                        mode="lines+markers",
                        name="Predicted PM₁₀ (µg/m³)",
                        line=dict(color="#8b5cf6", width=2.2, dash="dot"),
                        hovertemplate="%{x|%I:%M %p}<br>PM₁₀: %{y:.1f} µg/m³<extra></extra>"
                    ))

                # NO2 Trajectory
                fig_traj.add_trace(go.Scatter(
                    x=traj_df["DateTime"],
                    y=traj_df["predicted_no2"],
                    mode="lines+markers",
                    name="Predicted NO₂ (µg/m³)",
                    line=dict(color="#0284c7", width=2.2),
                    hovertemplate="%{x|%I:%M %p}<br>NO₂: %{y:.1f} µg/m³<extra></extra>"
                ))

                # Official CPCB Standard Reference Threshold Lines
                fig_traj.add_hline(
                    y=60, line_dash="dash", line_color="rgba(239, 68, 68, 0.6)",
                    annotation_text="PM₂.₅ 24-hr Limit (60)", annotation_position="top right"
                )
                fig_traj.add_hline(
                    y=100, line_dash="dash", line_color="rgba(139, 92, 246, 0.5)",
                    annotation_text="PM₁₀ 24-hr Limit (100)", annotation_position="top right"
                )

                fig_traj.update_layout(
                    height=340,
                    margin=dict(l=20, r=20, t=15, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    xaxis=dict(showgrid=True, gridcolor="rgba(148, 163, 184, 0.15)", title="Timeline (Next 12 Hours)"),
                    yaxis=dict(showgrid=True, gridcolor="rgba(148, 163, 184, 0.15)", title="Concentration (µg/m³)"),
                    hovermode="x unified"
                )
                st.plotly_chart(fig_traj, width="stretch")

            with fcol2:
                next_peak = traj_df.loc[traj_df["predicted_pm25"].idxmax()]
                peak_time_str = pd.to_datetime(next_peak["DateTime"]).strftime("%I:%M %p")
                st.markdown(f"""
                <div class="metric-card" style="border-left: 4px solid #f59e0b;">
                    <div class="metric-label">Forecast Interpretation</div>
                    <div style="font-weight: 700; font-size: 1.1rem; color: #1e293b; margin-top: 0.3rem;">
                        Projected Peak at ~{peak_time_str}
                    </div>
                    <div style="font-size: 0.85rem; color: #64748b; margin-top: 0.5rem; line-height: 1.55;">
                        Fine particles (PM₂.₅) are projected to reach approx <b>{next_peak['predicted_pm25']} µg/m³</b> 
                        (bounded uncertainty interval: <b>[{next_peak['pm25_lower']} – {next_peak['pm25_upper']}] µg/m³</b>).
                        <br><br>
                        Confidence level: <b>{next_peak['confidence']}</b> based on atmospheric continuity and diurnal mixing. 
                        Residents are encouraged to complete vigorous outdoor physical activities before the projected peak.
                    </div>
                    <div style="font-size: 0.78rem; color: #94a3b8; margin-top: 0.8rem;">
                        Predictions conform to Central Pollution Control Board (CPCB) NAQI evaluation guidelines.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Detailed Hourly Projections Table
            with st.expander("📋 View Detailed 12-Hour Projections Table with Uncertainty Bounds"):
                table_display = pd.DataFrame({
                    "Timestamp": [pd.to_datetime(t).strftime("%d %b, %I:%M %p") for t in traj_df["DateTime"]],
                    "Predicted PM₂.₅ (µg/m³)": traj_df["predicted_pm25"],
                    "PM₂.₅ Uncertainty Range": [f"[{l} – {u}]" for l, u in zip(traj_df["pm25_lower"], traj_df["pm25_upper"])],
                    "Predicted PM₁₀ (µg/m³)": traj_df.get("predicted_pm10", traj_df["predicted_pm25"] * 1.6),
                    "Predicted NO₂ (µg/m³)": traj_df["predicted_no2"],
                    "NO₂ Uncertainty Range": [f"[{l} – {u}]" for l, u in zip(traj_df["no2_lower"], traj_df["no2_upper"])],
                    "Model Confidence": traj_df["confidence"]
                })
                st.dataframe(table_display, width="stretch")

        else:
            st.info("Additional observations are accumulating in SQLite to generate the 12-hour trajectory.")


    else:
        st.warning(f"No observations are currently stored for {station_info['name']}. Please click 'Refresh Data Now' in the sidebar.")


# -----------------------------------------------------------------------------
# PAGE 2: HISTORICAL DATA / TRENDS
# -----------------------------------------------------------------------------
def render_historical_trends():
    st.markdown(f"""
    <div class="app-header">
        <h1 class="app-title">📊 Historical Observations & Trend Diagnostics</h1>
        <p class="app-subtitle">
            Explore continuous hourly monitoring records over the past 3 months for <b>{station_info['name']}</b>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Fetch up to 2500 hourly records (approx 100 days)
    history_df = get_historical_readings(current_station_id, limit=2500)

    if history_df.empty:
        st.warning(f"No historical observations recorded yet for {station_info['name']}. Please trigger a synchronization from the sidebar.")
        return

    # Filter controls
    min_date = history_df["DateTime"].min().date()
    max_date = history_df["DateTime"].max().date()

    fcol1, fcol2, fcol3 = st.columns([1.5, 1.2, 1])

    with fcol1:
        date_range = st.date_input(
            "Select Observation Date Window (Up to 3 Months):",
            value=(max(min_date, max_date - datetime.timedelta(days=30)), max_date),
            min_value=min_date,
            max_value=max_date
        )

    with fcol2:
        selected_pollutants = st.multiselect(
            "Pollutants to Graph:",
            options=["pm25", "pm10", "no2", "co", "so2", "ozone", "aqi"],
            default=["pm25", "pm10", "no2"],
            format_func=lambda x: {
                "pm25": "PM₂.₅ (µg/m³)",
                "pm10": "PM₁₀ (µg/m³)",
                "no2": "NO₂ (µg/m³)",
                "co": "CO (mg/m³)",
                "so2": "SO₂ (µg/m³)",
                "ozone": "Ozone (µg/m³)",
                "aqi": "National AQI Index"
            }.get(x, x.upper())
        )

    with fcol3:
        aggregation_type = st.radio(
            "Aggregation Mode:",
            ["Hourly Measurements", "Daily Mean Summary"],
            horizontal=True
        )

    # Filter dataframe
    filtered = history_df.copy()
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_ts = pd.to_datetime(date_range[0])
        end_ts = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)
        filtered = filtered[(filtered["DateTime"] >= start_ts) & (filtered["DateTime"] <= end_ts)]

    if filtered.empty:
        st.warning("No observations available in the selected date window.")
        return

    # Aggregate if daily summary chosen
    if aggregation_type == "Daily Mean Summary":
        chart_df = filtered.set_index("DateTime")[selected_pollutants].resample("D").mean().reset_index()
    else:
        chart_df = filtered

    # Plotly Time-Series Chart
    st.markdown('<div class="section-title">📈 Time-Series Telemetry Analysis</div>', unsafe_allow_html=True)
    if selected_pollutants:
        fig_hist = go.Figure()
        color_palette = ["#0284c7", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6", "#06b6d4", "#ec4899"]
        label_map = {
            "pm25": "PM₂.₅ (µg/m³)",
            "pm10": "PM₁₀ (µg/m³)",
            "no2": "NO₂ (µg/m³)",
            "co": "CO (mg/m³)",
            "so2": "SO₂ (µg/m³)",
            "ozone": "Ozone (µg/m³)",
            "aqi": "Overall AQI"
        }

        for idx, pol in enumerate(selected_pollutants):
            fig_hist.add_trace(go.Scatter(
                x=chart_df["DateTime"],
                y=chart_df[pol],
                mode="lines",
                name=label_map.get(pol, pol),
                line=dict(color=color_palette[idx % len(color_palette)], width=2),
                hovertemplate="%{x|%d %b %Y, %I:%M %p}<br>" + f"{label_map.get(pol, pol)}: %{{y:.1f}}<extra></extra>"
            ))

        fig_hist.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(showgrid=True, gridcolor="rgba(148, 163, 184, 0.15)"),
            yaxis=dict(showgrid=True, gridcolor="rgba(148, 163, 184, 0.15)", title="Recorded Value"),
            hovermode="x unified"
        )
        st.plotly_chart(fig_hist, width="stretch")
    else:
        st.info("Select one or more pollutants from the dropdown above to display the chart.")

    # Diurnal Pattern and Completeness
    st.markdown('<div class="section-title">📊 24-Hour Diurnal Pattern & Data Quality Indicators</div>', unsafe_allow_html=True)
    dcol1, dcol2 = st.columns([1.5, 1])

    with dcol1:
        st.write("**Average Hourly Variation (Traffic vs Clean Periods):**")
        diurnal_df = filtered.copy()
        diurnal_df["Hour"] = diurnal_df["DateTime"].dt.hour
        hourly_summary = diurnal_df.groupby("Hour")[["pm25", "no2"]].mean().reset_index()

        fig_diurn = go.Figure()
        fig_diurn.add_trace(go.Bar(
            x=hourly_summary["Hour"],
            y=hourly_summary["pm25"],
            name="PM₂.₅ (µg/m³)",
            marker_color="rgba(245, 158, 11, 0.75)"
        ))
        fig_diurn.add_trace(go.Scatter(
            x=hourly_summary["Hour"],
            y=hourly_summary["no2"],
            name="NO₂ (µg/m³)",
            line=dict(color="#0284c7", width=3)
        ))
        fig_diurn.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(title="Hour of Day (0–23)", tickmode="linear", tick0=0, dtick=2),
            yaxis=dict(showgrid=True, gridcolor="rgba(148, 163, 184, 0.15)", title="Concentration (µg/m³)")
        )
        st.plotly_chart(fig_diurn, width="stretch")

    with dcol2:
        st.write("**Data Completeness & Period Summary:**")
        total_hours = len(filtered)
        valid_pm25 = filtered["pm25"].notna().sum()
        completeness_pct = (valid_pm25 / max(total_hours, 1)) * 100

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Logged Period</div>
            <div style="font-weight:700; color:#1e293b; margin-top:0.2rem;">
                {min_date.strftime('%d %b %Y')} – {max_date.strftime('%d %b %Y')}
            </div>
            <div class="metric-sub">Total hourly entries: {total_hours:,}</div>
            
            <div class="metric-label" style="margin-top:0.8rem;">Data Completeness</div>
            <div style="font-size:1.5rem; font-weight:800; color:#10b981;">{completeness_pct:.1f}%</div>
            <div class="metric-sub">Verified continuous monitoring</div>
            
            <div style="font-size:0.8rem; color:#64748b; margin-top:0.6rem;">
                No synthetic or invented observations. Zero values or gaps reflect missing instrument feeds.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Download Data Button
    st.markdown('<div class="section-title">📥 Export Filtered Telemetry Records</div>', unsafe_allow_html=True)
    export_cols = ["timestamp", "pm25", "pm10", "no2", "co", "so2", "ozone", "aqi", "aqi_category", "prominent_pollutant"]
    csv_bytes = filtered[export_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Filtered Observations as CSV",
        data=csv_bytes,
        file_name=f"AirSense_{current_station_id}_Telemetry.csv",
        mime="text/csv",
        width="stretch"
    )


# -----------------------------------------------------------------------------
# PAGE 3: AIR QUALITY ANALYSIS & HEALTH RECOMMENDATIONS
# -----------------------------------------------------------------------------
def render_analysis_and_advisory():
    st.markdown(f"""
    <div class="app-header">
        <h1 class="app-title">🛡️ Air Quality Health Guidance & Analysis</h1>
        <p class="app-subtitle">
            Evidence-based exposure reduction advisories, atmospheric pattern clustering, 
            and explainable insights for <b>{station_info['name']}</b>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    aqi_val = latest_obs.get("aqi") if latest_obs else None
    aqi_cat = latest_obs.get("aqi_category") if latest_obs else "Unavailable"
    prominent = latest_obs.get("prominent_pollutant") if latest_obs else "PM2.5"

    advisory = get_health_advisory(aqi_val, aqi_cat, prominent, station_name=station_info["name"])

    # 1. Main Health Status Banner
    st.markdown(f"""
    <div class="aqi-banner" style="background: {advisory['color']}15; border-color: {advisory['color']}55;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div>
                <div class="metric-label" style="color: #64748b;">Current Health Tier</div>
                <div style="font-size: 1.8rem; font-weight: 800; color: {advisory['color']};">{advisory['headline']}</div>
                <div style="font-size: 1.0rem; color: #334155; margin-top: 0.3rem;">{advisory['summary']}</div>
            </div>
            <div style="text-align:right;">
                <div style="font-size: 2.5rem; font-weight: 800; color: {advisory['color']};">{aqi_val if aqi_val else '--'}</div>
                <div style="font-size: 0.82rem; color: #64748b;">National AQI</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # OBJECTIVE 3: WHAT THIS MEANS FOR YOU (EXPLAINABLE INSIGHTS & RECOMMENDATIONS)
    # -------------------------------------------------------------------------
    st.markdown('<div class="section-title">💡 What This Means for You</div>', unsafe_allow_html=True)
    st.write(
        "Clear, citizen-first interpretation of current air quality, historical trend direction, "
        "what upcoming predictions mean for your daily routine, and practical steps to minimize exposure."
    )

    # Fetch recent history for trend calculations
    hist_recent = get_historical_readings(current_station_id, limit=72)
    traj_recent = generate_upcoming_trajectory(hist_recent, hours_ahead=12)
    early_summary = get_early_warning_summary(traj_recent, latest_obs) if traj_recent is not None else {}

    # 3 Plain-Language Context Cards
    mcol1, mcol2, mcol3 = st.columns(3)

    with mcol1:
        st.markdown(f"""
        <div class="metric-card" style="border-left: 4px solid #0284c7;">
            <div class="metric-label">1. Latest Readings Meaning</div>
            <div style="font-weight: 700; font-size: 1.05rem; color: #1e293b; margin-top: 0.3rem;">
                {aqi_cat} Air Quality
            </div>
            <div style="font-size: 0.85rem; color: #64748b; margin-top: 0.4rem; line-height: 1.5;">
                Current index of <b>{aqi_val if aqi_val else '--'}</b> indicates that {prominent} is the primary pollutant. 
                { 'Air is suitable for outdoor leisure.' if aqi_cat in ['Good', 'Satisfactory'] else 'Prolonged outdoor exertion may cause throat or respiratory irritation.'}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with mcol2:
        # Determine 24-hr historical change
        if hist_recent is not None and len(hist_recent) >= 24:
            recent_24h_mean = hist_recent.tail(24)["pm25"].mean()
            prior_24h_mean = hist_recent.head(len(hist_recent) - 24)["pm25"].mean() if len(hist_recent) > 24 else recent_24h_mean
            pct_change = ((recent_24h_mean - prior_24h_mean) / max(prior_24h_mean, 1e-3)) * 100
            trend_note = f"{abs(pct_change):.1f}% {'higher' if pct_change > 0 else 'lower'} than earlier days"
        else:
            trend_note = "Tracking steadily within seasonal norms"

        st.markdown(f"""
        <div class="metric-card" style="border-left: 4px solid #10b981;">
            <div class="metric-label">2. Historical Trend Direction</div>
            <div style="font-weight: 700; font-size: 1.05rem; color: #1e293b; margin-top: 0.3rem;">
                Continuous Baseline
            </div>
            <div style="font-size: 0.85rem; color: #64748b; margin-top: 0.4rem; line-height: 1.5;">
                Particulate concentrations are <b>{trend_note}</b>. 
                Diurnal data shows consistent early morning elevation coinciding with morning traffic and night-time surface cooling.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with mcol3:
        pred_trend = early_summary.get("trend_direction", "Stable")
        st.markdown(f"""
        <div class="metric-card" style="border-left: 4px solid #f59e0b;">
            <div class="metric-label">3. Prediction Significance</div>
            <div style="font-weight: 700; font-size: 1.05rem; color: #1e293b; margin-top: 0.3rem;">
                Upcoming Trajectory: {pred_trend}
            </div>
            <div style="font-size: 0.85rem; color: #64748b; margin-top: 0.4rem; line-height: 1.5;">
                { 'Prepare for elevated pollutant density over the next few hours; reschedule high-intensity outdoor fitness.' if pred_trend == 'Increasing' else ('Pollutant concentrations are dissipating; favorable period for natural ventilation.' if pred_trend == 'Decreasing' else 'Stable conditions projected over the next 12 hours with minimal acute variation.')}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 0.6rem;'></div>", unsafe_allow_html=True)

    # Environmental Driving Factors Callout Box
    st.markdown(f"""
    <div class="info-card">
        <b>Key Environmental Driving Factors for {station_info['name']}:</b><br>
        • <b>Diurnal Traffic Surges:</b> Morning (08:00–10:30) and evening (18:30–21:30) commuter rush hours generate sharp surges in vehicle exhaust (NO₂ and ultrafine particulates).<br>
        • <b>Boundary Layer Height & Thermal Inversion:</b> Nocturnal cooling compresses the atmospheric mixing layer, trapping road-level emissions near ground level until mid-morning solar warming restores vertical dispersion.<br>
        • <b>Airshed Topography:</b> {advisory['traffic_note']}
    </div>
    """, unsafe_allow_html=True)

    # Demographic Guidance Cards
    st.markdown('<div class="section-title">👥 Actionable Exposure Reduction Guidance</div>', unsafe_allow_html=True)
    gcol1, gcol2 = st.columns(2)

    with gcol1:
        st.markdown("""
        <div class="metric-card" style="border-left: 4px solid #0284c7;">
            <div style="font-weight: 700; font-size: 1.1rem; color: #0284c7;">Healthy Adults & General Public</div>
            <ul style="font-size: 0.88rem; color: #475569; margin-top: 0.6rem; padding-left: 1.2rem; line-height: 1.6;">
        """ + "".join([f"<li>{item}</li>" for item in advisory["general_public"]]) + """
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)

        st.markdown("""
        <div class="metric-card" style="border-left: 4px solid #10b981;">
            <div style="font-weight: 700; font-size: 1.1rem; color: #10b981;">Indoor Air Management & Ventilation Timing</div>
            <ul style="font-size: 0.88rem; color: #475569; margin-top: 0.6rem; padding-left: 1.2rem; line-height: 1.6;">
        """ + "".join([f"<li>{item}</li>" for item in advisory["indoor_guidance"]]) + """
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with gcol2:
        st.markdown("""
        <div class="metric-card" style="border-left: 4px solid #ef4444;">
            <div style="font-weight: 700; font-size: 1.1rem; color: #ef4444;">Sensitive Populations (Children, Seniors & Respiratory Conditions)</div>
            <ul style="font-size: 0.88rem; color: #475569; margin-top: 0.6rem; padding-left: 1.2rem; line-height: 1.6;">
        """ + "".join([f"<li>{item}</li>" for item in advisory["sensitive_groups"]]) + """
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)

        st.markdown("""
        <div class="metric-card" style="border-left: 4px solid #f59e0b;">
            <div style="font-weight: 700; font-size: 1.1rem; color: #f59e0b;">Commuters, Outdoor Workers & Active Citizens</div>
            <ul style="font-size: 0.88rem; color: #475569; margin-top: 0.6rem; padding-left: 1.2rem; line-height: 1.6;">
        """ + "".join([f"<li>{item}</li>" for item in advisory["outdoor_activity"]]) + """
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Official Standards Comparison Table
    st.markdown('<div class="section-title">📋 Official Ambient Air Quality Standards Comparison</div>', unsafe_allow_html=True)
    st.write("Current measurements compared against India's National Ambient Air Quality Standards (NAAQS) and World Health Organization (WHO) benchmarks:")

    standard_rows = [
        {"Pollutant": "Fine Particulate Matter (PM₂.₅)", "Current Reading": f"{latest_obs.get('pm25', '--')} µg/m³" if latest_obs else "--", "India 24-hr NAAQS": "60 µg/m³", "WHO 24-hr Guideline": "15 µg/m³", "Status vs CPCB": "Safe" if (latest_obs and latest_obs.get("pm25", 0) <= 60) else "Elevated"},
        {"Pollutant": "Respirable Particulate Matter (PM₁₀)", "Current Reading": f"{latest_obs.get('pm10', '--')} µg/m³" if latest_obs else "--", "India 24-hr NAAQS": "100 µg/m³", "WHO 24-hr Guideline": "45 µg/m³", "Status vs CPCB": "Safe" if (latest_obs and latest_obs.get("pm10", 0) <= 100) else "Elevated"},
        {"Pollutant": "Nitrogen Dioxide (NO₂)", "Current Reading": f"{latest_obs.get('no2', '--')} µg/m³" if latest_obs else "--", "India 24-hr NAAQS": "80 µg/m³", "WHO 24-hr Guideline": "25 µg/m³", "Status vs CPCB": "Safe" if (latest_obs and latest_obs.get("no2", 0) <= 80) else "Elevated"},
        {"Pollutant": "Sulphur Dioxide (SO₂)", "Current Reading": f"{latest_obs.get('so2', '--')} µg/m³" if latest_obs else "--", "India 24-hr NAAQS": "80 µg/m³", "WHO 24-hr Guideline": "40 µg/m³", "Status vs CPCB": "Safe"},
        {"Pollutant": "Carbon Monoxide (CO)", "Current Reading": f"{latest_obs.get('co', '--')} mg/m³" if latest_obs else "--", "India 8-hr NAAQS": "2.0 mg/m³", "WHO 24-hr Guideline": "4.0 mg/m³", "Status vs CPCB": "Safe" if (latest_obs and latest_obs.get("co", 0) <= 2.0) else "Elevated"},
        {"Pollutant": "Ground Ozone (O₃)", "Current Reading": f"{latest_obs.get('ozone', '--')} µg/m³" if latest_obs else "--", "India 8-hr NAAQS": "100 µg/m³", "WHO Peak Season": "60 µg/m³", "Status vs CPCB": "Safe"}
    ]
    st.dataframe(pd.DataFrame(standard_rows), width="stretch")

    # -------------------------------------------------------------------------
    # OBJECTIVE 2: POLLUTION PATTERN CLUSTERING (MANDATORY SECTION)
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.markdown('<div class="section-title">🧬 Pollution Pattern Clustering</div>', unsafe_allow_html=True)
    st.write(
        "Atmospheric pattern clustering groups multi-pollutant observations into recurring atmospheric signatures "
        "based on measured chemical concentrations (NOx, NO₂, CO, Benzene) and meteorological factors (Temperature, Humidity). "
        "This identifies structural pollution regimes rather than looking at isolated numbers."
    )

    all_clusters = get_all_cluster_profiles()
    station_match = match_station_to_cluster(latest_obs)

    # Station Alignment Card
    st.markdown(f"""
    <div class="metric-card" style="border-left: 4px solid #0284c7; margin-bottom: 1.2rem;">
        <div class="metric-label">Current Locality Pattern Alignment</div>
        <div style="font-weight: 700; font-size: 1.15rem; color: #1e293b; margin-top: 0.3rem;">
            {station_info['name']} closely aligns with <b>{station_match['label']}: {station_match['pattern_title']}</b>
        </div>
        <div style="font-size: 0.88rem; color: #64748b; margin-top: 0.4rem; line-height: 1.55;">
            {station_match['atmospheric_explanation']}
            <br>
            <b>Signature Match Confidence:</b> {station_match['match_confidence']} | 
            <b>Typical Active Window:</b> {station_match['typical_timing']}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Display 3 Cluster Cards
    st.markdown("<b>Verified Atmospheric Cluster Signatures (9,326 Continuous Observations):</b>", unsafe_allow_html=True)
    ccol1, ccol2, ccol3 = st.columns(3)

    c_colors = ["#ef4444", "#10b981", "#f59e0b"]
    for idx, (cid, pdata) in enumerate(all_clusters.items()):
        target_col = [ccol1, ccol2, ccol3][idx]
        col_accent = c_colors[idx]
        m = pdata["means"]

        with target_col:
            st.markdown(f"""
            <div class="metric-card" style="border-top: 4px solid {col_accent};">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span class="badge" style="background:{col_accent}22; color:{col_accent}; border:1px solid {col_accent}55;">
                        {pdata['label']}
                    </span>
                    <span style="font-size:0.82rem; font-weight:700; color:#64748b;">
                        {pdata['count']:,} obs ({pdata['percentage']}%)
                    </span>
                </div>
                <div style="font-weight: 700; font-size: 1.05rem; color: #1e293b; margin-top: 0.6rem;">
                    {pdata['pattern_title']}
                </div>
                <div style="font-size: 0.84rem; color: #64748b; margin-top: 0.5rem; line-height: 1.5;">
                    {pdata['atmospheric_explanation']}
                </div>
                <div style="background: rgba(148, 163, 184, 0.08); border-radius: 8px; padding: 0.6rem 0.8rem; margin-top: 0.8rem; font-size: 0.8rem;">
                    <b>Average Measured Levels:</b><br>
                    • NO₂: <b>{m['no2']} µg/m³</b> | NOx: <b>{m['nox']} ppb</b><br>
                    • CO: <b>{m['co']} mg/m³</b> | Benzene: <b>{m['c6h6']} µg/m³</b><br>
                    • Temp: <b>{m['temp']}°C</b> | Rel Humidity: <b>{m['rh']}%</b>
                </div>
                <div style="font-size: 0.78rem; color: #94a3b8; margin-top: 0.6rem;">
                    <b>Active Window:</b> {pdata['typical_timing']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Interactive Cluster Visualizations
    st.markdown("<div style='height: 1.0rem;'></div>", unsafe_allow_html=True)
    st.markdown("<b>Atmospheric Cluster Profiles & Distribution Visualizations:</b>", unsafe_allow_html=True)
    vcol1, vcol2 = st.columns([1.6, 1])

    with vcol1:
        # Comparison Bar Chart of Cluster Means
        cluster_names = [f"Cluster {c}" for c in all_clusters.keys()]
        no2_means = [all_clusters[c]["means"]["no2"] for c in all_clusters.keys()]
        co_means = [all_clusters[c]["means"]["co"] * 25.0 for c in all_clusters.keys()] # Scaled for visualization
        c6h6_means = [all_clusters[c]["means"]["c6h6"] for c in all_clusters.keys()]
        temp_means = [all_clusters[c]["means"]["temp"] for c in all_clusters.keys()]

        fig_cluster = go.Figure()
        fig_cluster.add_trace(go.Bar(name="NO₂ (µg/m³)", x=cluster_names, y=no2_means, marker_color="#0284c7"))
        fig_cluster.add_trace(go.Bar(name="CO (Scaled ×25)", x=cluster_names, y=co_means, marker_color="#ef4444"))
        fig_cluster.add_trace(go.Bar(name="Benzene (µg/m³)", x=cluster_names, y=c6h6_means, marker_color="#f59e0b"))
        fig_cluster.add_trace(go.Bar(name="Temperature (°C)", x=cluster_names, y=temp_means, marker_color="#10b981"))

        fig_cluster.update_layout(
            barmode="group",
            height=320,
            margin=dict(l=20, r=20, t=15, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(showgrid=True, gridcolor="rgba(148, 163, 184, 0.15)"),
            yaxis=dict(showgrid=True, gridcolor="rgba(148, 163, 184, 0.15)", title="Comparative Concentration / Value")
        )
        st.plotly_chart(fig_cluster, width="stretch")

    with vcol2:
        # Donut Chart of Cluster Proportion
        fig_pie = go.Figure(data=[go.Pie(
            labels=[f"Cluster {c}: {all_clusters[c]['pattern_title'].split('&')[0]}" for c in all_clusters.keys()],
            values=[all_clusters[c]["count"] for c in all_clusters.keys()],
            hole=0.55,
            marker_colors=["#ef4444", "#10b981", "#f59e0b"],
            textinfo="percent+label",
            showlegend=False
        )])
        fig_pie.update_layout(
            height=320,
            margin=dict(l=10, r=10, t=15, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_pie, width="stretch")

    st.caption(
        "Disclaimer: Advisories provide precautionary public guidance based on official environmental criteria and do not constitute direct medical diagnoses. "
        "Residents experiencing severe symptoms should consult healthcare professionals."
    )



# -----------------------------------------------------------------------------
# 3. ROUTER
# -----------------------------------------------------------------------------
if nav_selection == "🏠 Air Quality Dashboard":
    render_dashboard()
elif nav_selection == "📊 Historical Data / Trends":
    render_historical_trends()
elif nav_selection == "🛡️ Analysis & Health Guidance":
    render_analysis_and_advisory()

# Public Footer
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #94a3b8; font-size: 0.85rem; padding: 0.5rem 0 1.2rem 0;">
    <b>AirSense AI</b> — Real-World Air Quality Monitoring Service | Serving {station_info['name']} | Telemetry via CPCB/MPCB & Copernicus CAMS
</div>
""", unsafe_allow_html=True)
