"""
AirSense AI - Real-World Air Quality Data Ingestion Client.

Integrates with verified atmospheric monitoring services (Copernicus CAMS & Open-Meteo Air Quality),
transforms pollutant units to standard Indian CPCB definitions, calculates official NAQI,
and saves historical and current observations into the persistent SQLite store.
"""

import datetime
import requests
import pandas as pd
from typing import Dict, Any, Optional, Tuple

from data.stations import get_station_by_id
from data.database import save_readings, get_latest_reading, get_historical_readings, log_sync
from analysis.aqi_calc import calculate_overall_aqi

API_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
DEFAULT_TIMEOUT_SEC = 12


def fetch_and_sync_station_data(station_id: str, past_days: int = 92) -> Tuple[bool, str, int]:
    """
    Fetch up to 3 months of real-world hourly atmospheric data for a station,
    compute Indian CPCB NAQI, and persist to SQLite with duplicate prevention.
    Returns: (success: bool, message: str, count_saved: int)
    """
    station = get_station_by_id(station_id)
    if not station:
        return False, f"Unknown station '{station_id}'.", 0

    params = {
        "latitude": station["latitude"],
        "longitude": station["longitude"],
        "hourly": [
            "pm10",
            "pm2_5",
            "carbon_monoxide",
            "nitrogen_dioxide",
            "sulphur_dioxide",
            "ozone"
        ],
        "past_days": past_days,
        "timezone": "Asia/Kolkata"
    }

    try:
        response = requests.get(API_URL, params=params, timeout=DEFAULT_TIMEOUT_SEC)
        if response.status_code != 200:
            err_msg = f"API returned HTTP {response.status_code}"
            log_sync(station_id, "ERROR", 0, err_msg)
            return False, err_msg, 0

        payload = response.json()
        hourly_data = payload.get("hourly", {})
        times = hourly_data.get("time", [])

        if not times:
            err_msg = "API response did not contain hourly telemetry arrays."
            log_sync(station_id, "ERROR", 0, err_msg)
            return False, err_msg, 0

        # Construct records
        pm10_list = hourly_data.get("pm10", [])
        pm25_list = hourly_data.get("pm2_5", [])
        co_list = hourly_data.get("carbon_monoxide", [])
        no2_list = hourly_data.get("nitrogen_dioxide", [])
        so2_list = hourly_data.get("sulphur_dioxide", [])
        o3_list = hourly_data.get("ozone", [])

        records = []
        for i, ts in enumerate(times):
            # Carbon monoxide from API is in ug/m3, convert to mg/m3
            raw_co = co_list[i] if i < len(co_list) else None
            co_mg = (raw_co / 1000.0) if (raw_co is not None and raw_co > 50) else raw_co

            pol_dict = {
                "pm25": pm25_list[i] if i < len(pm25_list) else None,
                "pm10": pm10_list[i] if i < len(pm10_list) else None,
                "no2": no2_list[i] if i < len(no2_list) else None,
                "so2": so2_list[i] if i < len(so2_list) else None,
                "co": co_mg,
                "ozone": o3_list[i] if i < len(o3_list) else None
            }

            aqi_val, aqi_cat, aqi_color, aqi_adv, sub_indices = calculate_overall_aqi(pol_dict)
            prominent = max(sub_indices, key=sub_indices.get) if sub_indices else "None"

            records.append({
                "timestamp": ts,
                "pm25": pol_dict["pm25"],
                "pm10": pol_dict["pm10"],
                "no2": pol_dict["no2"],
                "co": pol_dict["co"],
                "so2": pol_dict["so2"],
                "ozone": pol_dict["ozone"],
                "aqi": aqi_val,
                "aqi_category": aqi_cat,
                "prominent_pollutant": prominent.upper() if prominent else "None"
            })

        df_sync = pd.DataFrame(records)
        saved = save_readings(station_id, df_sync, source_label=station["provider"])
        log_sync(station_id, "SUCCESS", saved, f"Fetched {len(times)} records.")
        return True, f"Successfully refreshed {saved} observations.", saved

    except requests.exceptions.Timeout:
        err_msg = "Request to monitoring API timed out (server busy). Preserving stored database cache."
        log_sync(station_id, "ERROR", 0, err_msg)
        return False, err_msg, 0
    except Exception as e:
        err_msg = f"Sync exception: {str(e)}"
        log_sync(station_id, "ERROR", 0, err_msg)
        return False, err_msg, 0


def get_station_telemetry(station_id: str, force_refresh: bool = False) -> Dict[str, Any]:
    """
    Retrieve live and recent telemetry for a station.
    If database has no records or force_refresh is True, triggers an ingestion.
    Returns structured dictionary with current observation, data status, attribution, and station metadata.
    """
    station = get_station_by_id(station_id)
    latest = get_latest_reading(station_id)

    # If no data exists or force refresh requested, fetch from API
    if latest is None or force_refresh:
        success, msg, count = fetch_and_sync_station_data(station_id)
        latest = get_latest_reading(station_id)

    # Determine status
    if latest is None:
        status_label = "Unavailable"
        status_color = "#94a3b8"
    else:
        # Check recency
        try:
            obs_dt = pd.to_datetime(latest["timestamp"])
            age_hours = (datetime.datetime.now() - obs_dt).total_seconds() / 3600.0
            if age_hours <= 3.0:
                status_label = "Live"
                status_color = "#10b981"
            elif age_hours <= 24.0:
                status_label = "Recently Updated"
                status_color = "#0284c7"
            else:
                status_label = "Historical Cache"
                status_color = "#f59e0b"
        except Exception:
            status_label = "Recently Updated"
            status_color = "#0284c7"

    return {
        "station": station,
        "latest": latest,
        "status_label": status_label,
        "status_color": status_color
    }
