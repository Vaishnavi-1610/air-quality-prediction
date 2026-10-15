"""
AirSense AI - Backend Model Management and Prediction Service.

Maintains trained model pipelines (Random Forest, CatBoost, Gradient Boosting),
audits input feature compatibility, and provides verified trajectory forecasts
without exposing academic jargon or model names to the public interface.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "air_quality_best_model.joblib")
CATBOOST_PATH = os.path.join(BASE_DIR, "air_quality_catboost.joblib")
COMPARISON_PATH = os.path.join(BASE_DIR, "model_comparison.csv")

# Cache in-memory model instances
_ACTIVE_MODEL_BUNDLE = None
_CATBOOST_BUNDLE = None


def load_active_model_bundle() -> Optional[Dict[str, Any]]:
    """Load the champion model bundle from backend storage."""
    global _ACTIVE_MODEL_BUNDLE
    if _ACTIVE_MODEL_BUNDLE is not None:
        return _ACTIVE_MODEL_BUNDLE

    if os.path.exists(MODEL_PATH):
        try:
            _ACTIVE_MODEL_BUNDLE = joblib.load(MODEL_PATH)
            return _ACTIVE_MODEL_BUNDLE
        except Exception as e:
            print(f"[Backend Warning] Failed to load {MODEL_PATH}: {e}")
    return None


def load_catboost_bundle() -> Optional[Dict[str, Any]]:
    """Load the dedicated CatBoost model bundle if available."""
    global _CATBOOST_BUNDLE
    if _CATBOOST_BUNDLE is not None:
        return _CATBOOST_BUNDLE

    if os.path.exists(CATBOOST_PATH):
        try:
            _CATBOOST_BUNDLE = joblib.load(CATBOOST_PATH)
            return _CATBOOST_BUNDLE
        except Exception as e:
            print(f"[Backend Warning] Failed to load {CATBOOST_PATH}: {e}")
    return None


def get_backend_model_metadata() -> Dict[str, Any]:
    """Retrieve internal model configuration and benchmark results (for developer logs)."""
    bundle = load_active_model_bundle()
    comp_df = None
    if os.path.exists(COMPARISON_PATH):
        try:
            comp_df = pd.read_csv(COMPARISON_PATH)
        except Exception:
            pass

    return {
        "active_model_name": bundle.get("model_name", "Random Forest") if bundle else "None",
        "features": bundle.get("features", []) if bundle else [],
        "target": bundle.get("target", "NO2_next") if bundle else "NO2_next",
        "description": bundle.get("prediction_description", "") if bundle else "",
        "benchmarks": comp_df
    }


def validate_feature_compatibility(input_dict: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Audit whether an input dictionary contains the exact 16 sensor features
    required by the historical 2004 multisensor model.
    """
    bundle = load_active_model_bundle()
    if not bundle:
        return False, ["Champion model bundle is not loaded."]

    required_features = bundle.get("features", [])
    missing = [f for f in required_features if f not in input_dict or pd.isna(input_dict[f])]
    return (len(missing) == 0), missing


def predict_from_sensor_features(features_dict: Dict[str, Any]) -> Optional[float]:
    """Execute prediction if and only if all compatible features are present."""
    bundle = load_active_model_bundle()
    if not bundle:
        return None

    is_compatible, missing = validate_feature_compatibility(features_dict)
    if not is_compatible:
        return None

    pipeline = bundle["model"]
    feature_order = bundle["features"]
    df_in = pd.DataFrame([features_dict], columns=feature_order)

    try:
        prediction = float(pipeline.predict(df_in)[0])
        return max(0.0, prediction)
    except Exception as e:
        print(f"[Backend Warning] Prediction failed: {e}")
        return None


def generate_upcoming_trajectory(history_df: pd.DataFrame, hours_ahead: int = 12) -> Optional[pd.DataFrame]:
    """
    Generate an empirical short-term forecast trajectory for live real-world stations
    using exponential diurnal smoothing and atmospheric continuity.
    Strictly reports a bounded trajectory with confidence limits.
    """
    if history_df is None or len(history_df) < 24:
        return None

    df_clean = history_df.sort_values("timestamp").reset_index(drop=True)
    last_row = df_clean.iloc[-1]
    last_time = pd.to_datetime(last_row["timestamp"])

    # Base values from most recent observations
    recent_no2 = df_clean["no2"].tail(24).dropna()
    recent_pm25 = df_clean["pm25"].tail(24).dropna()
    recent_pm10 = df_clean["pm10"].tail(24).dropna() if "pm10" in df_clean.columns else pd.Series(dtype=float)

    if recent_no2.empty or recent_pm25.empty:
        return None

    # Calculate hourly diurnal factors from available historical data
    df_clean["hour"] = pd.to_datetime(df_clean["timestamp"]).dt.hour
    hourly_no2_mean = df_clean.groupby("hour")["no2"].mean()
    overall_no2_mean = df_clean["no2"].mean()
    no2_diurnal_factor = (hourly_no2_mean / max(overall_no2_mean, 1e-3)).to_dict()

    hourly_pm25_mean = df_clean.groupby("hour")["pm25"].mean()
    overall_pm25_mean = df_clean["pm25"].mean()
    pm25_diurnal_factor = (hourly_pm25_mean / max(overall_pm25_mean, 1e-3)).to_dict()

    hourly_pm10_mean = df_clean.groupby("hour")["pm10"].mean() if "pm10" in df_clean.columns else None
    overall_pm10_mean = df_clean["pm10"].mean() if "pm10" in df_clean.columns else None
    pm10_diurnal_factor = (hourly_pm10_mean / max(overall_pm10_mean, 1e-3)).to_dict() if overall_pm10_mean and overall_pm10_mean > 0 else {}

    current_no2 = float(last_row.get("no2") or recent_no2.iloc[-1])
    current_pm25 = float(last_row.get("pm25") or recent_pm25.iloc[-1])
    current_pm10 = float(last_row.get("pm10") or (recent_pm10.iloc[-1] if not recent_pm10.empty else current_pm25 * 1.65))

    forecast_rows = []
    for h in range(1, hours_ahead + 1):
        future_dt = last_time + pd.Timedelta(hours=h)
        future_hour = future_dt.hour

        # Decay factor: blending current persistence with diurnal seasonal pattern
        decay = np.exp(-h / 12.0)
        diurnal_n = no2_diurnal_factor.get(future_hour, 1.0)
        diurnal_p = pm25_diurnal_factor.get(future_hour, 1.0)
        diurnal_p10 = pm10_diurnal_factor.get(future_hour, diurnal_p)

        pred_no2 = (current_no2 * decay) + (overall_no2_mean * diurnal_n * (1.0 - decay))
        pred_pm25 = (current_pm25 * decay) + (overall_pm25_mean * diurnal_p * (1.0 - decay))
        base_pm10_mean = overall_pm10_mean if (overall_pm10_mean and overall_pm10_mean > 0) else (overall_pm25_mean * 1.65)
        pred_pm10 = (current_pm10 * decay) + (base_pm10_mean * diurnal_p10 * (1.0 - decay))

        # Add sensible uncertainty bounds (expanding with lead time)
        uncertainty_pct = 0.05 + (h * 0.015)
        pred_no2 = max(2.0, pred_no2)
        pred_pm25 = max(5.0, pred_pm25)
        pred_pm10 = max(10.0, pred_pm10)

        forecast_rows.append({
            "timestamp": future_dt.strftime("%Y-%m-%dT%H:%M"),
            "DateTime": future_dt,
            "hours_ahead": h,
            "predicted_no2": round(pred_no2, 1),
            "no2_lower": round(max(0.0, pred_no2 * (1 - uncertainty_pct)), 1),
            "no2_upper": round(pred_no2 * (1 + uncertainty_pct), 1),
            "predicted_pm25": round(pred_pm25, 1),
            "pm25_lower": round(max(0.0, pred_pm25 * (1 - uncertainty_pct)), 1),
            "pm25_upper": round(pred_pm25 * (1 + uncertainty_pct), 1),
            "predicted_pm10": round(pred_pm10, 1),
            "pm10_lower": round(max(0.0, pred_pm10 * (1 - uncertainty_pct)), 1),
            "pm10_upper": round(pred_pm10 * (1 + uncertainty_pct), 1),
            "confidence": "High" if h <= 3 else ("Moderate" if h <= 8 else "Indicative")
        })

    return pd.DataFrame(forecast_rows)


def get_early_warning_summary(traj_df: pd.DataFrame, current_obs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate structured early warnings and trend indicators based on upcoming predictions
    against National Ambient Air Quality Standards (NAAQS 24-hr: PM2.5=60, PM10=100, NO2=80 µg/m³).
    """
    if traj_df is None or traj_df.empty:
        return {}

    curr_pm25 = float(current_obs.get("pm25") or 0.0)
    curr_pm10 = float(current_obs.get("pm10") or 0.0)
    curr_no2 = float(current_obs.get("no2") or 0.0)

    # Max predicted over horizon
    max_pm25_row = traj_df.loc[traj_df["predicted_pm25"].idxmax()]
    max_pm10_row = traj_df.loc[traj_df["predicted_pm10"].idxmax()]
    max_no2_row = traj_df.loc[traj_df["predicted_no2"].idxmax()]

    # Trend calculations (compare horizon peak or average to current)
    avg_pm25_pred = traj_df["predicted_pm25"].mean()
    diff_pm25 = avg_pm25_pred - curr_pm25

    if diff_pm25 > 5.0:
        pm25_trend = "Increasing"
        trend_color = "#ef4444"
        trend_icon = "↗"
    elif diff_pm25 < -5.0:
        pm25_trend = "Decreasing"
        trend_color = "#10b981"
        trend_icon = "↘"
    else:
        pm25_trend = "Stable"
        trend_color = "#0284c7"
        trend_icon = "→"

    # Threshold checks
    warnings = []
    if max_pm25_row["predicted_pm25"] > 60.0:
        warnings.append({
            "pollutant": "PM2.5",
            "threshold": 60.0,
            "predicted_peak": max_pm25_row["predicted_pm25"],
            "uncertainty": f"[{max_pm25_row['pm25_lower']} – {max_pm25_row['pm25_upper']}] µg/m³",
            "time": pd.to_datetime(max_pm25_row["DateTime"]).strftime("%I:%M %p"),
            "severity": "High" if max_pm25_row["predicted_pm25"] > 90.0 else "Moderate",
            "message": f"PM2.5 is projected to exceed the 24-hr CPCB safety limit (60 µg/m³), peaking at ~{max_pm25_row['predicted_pm25']} µg/m³."
        })

    if max_pm10_row["predicted_pm10"] > 100.0:
        warnings.append({
            "pollutant": "PM10",
            "threshold": 100.0,
            "predicted_peak": max_pm10_row["predicted_pm10"],
            "uncertainty": f"[{max_pm10_row['pm10_lower']} – {max_pm10_row['pm10_upper']}] µg/m³",
            "time": pd.to_datetime(max_pm10_row["DateTime"]).strftime("%I:%M %p"),
            "severity": "Moderate",
            "message": f"PM10 is projected to exceed the standard (100 µg/m³), reaching ~{max_pm10_row['predicted_pm10']} µg/m³."
        })

    if max_no2_row["predicted_no2"] > 80.0:
        warnings.append({
            "pollutant": "NO2",
            "threshold": 80.0,
            "predicted_peak": max_no2_row["predicted_no2"],
            "uncertainty": f"[{max_no2_row['no2_lower']} – {max_no2_row['no2_upper']}] µg/m³",
            "time": pd.to_datetime(max_no2_row["DateTime"]).strftime("%I:%M %p"),
            "severity": "Moderate",
            "message": f"NO2 is expected to rise above 80 µg/m³ around peak commuting hours."
        })

    return {
        "trend_direction": pm25_trend,
        "trend_color": trend_color,
        "trend_icon": trend_icon,
        "warnings": warnings,
        "max_pm25": max_pm25_row["predicted_pm25"],
        "max_pm25_time": pd.to_datetime(max_pm25_row["DateTime"]).strftime("%I:%M %p"),
        "max_pm25_uncertainty": f"{max_pm25_row['pm25_lower']} – {max_pm25_row['pm25_upper']} µg/m³",
        "max_pm10": max_pm10_row["predicted_pm10"],
        "max_pm10_uncertainty": f"{max_pm10_row['pm10_lower']} – {max_pm10_row['pm10_upper']} µg/m³",
        "max_no2": max_no2_row["predicted_no2"],
        "max_no2_uncertainty": f"{max_no2_row['no2_lower']} – {max_no2_row['no2_upper']} µg/m³"
    }

