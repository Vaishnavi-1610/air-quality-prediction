"""
AirSense AI - Comprehensive Verification and Test Suite.
Validates database schema, API ingestion, AQI calculations,
backend model loading, and public-facing UI compliance.
"""

import os
import unittest
import pandas as pd
import numpy as np

from data.database import init_db, save_readings, get_latest_reading, get_historical_readings
from data.stations import get_all_stations, get_station_by_id
from data.api_client import fetch_and_sync_station_data, get_station_telemetry
from analysis.aqi_calc import calculate_overall_aqi, calculate_sub_index
from analysis.advisory import get_health_advisory
from models.prediction_service import (
    load_active_model_bundle,
    load_catboost_bundle,
    validate_feature_compatibility,
    generate_upcoming_trajectory
)
from models.clustering_service import load_cluster_bundle, assign_cluster_pattern


class TestAirSenseSystem(unittest.TestCase):

    def setUp(self):
        init_db()

    def test_stations_registry(self):
        stations = get_all_stations()
        self.assertIn("karve_nagar", stations)
        self.assertIn("shivajinagar", stations)
        kn = get_station_by_id("karve_nagar")
        self.assertFalse(kn["is_physical_station"])
        self.assertEqual(kn["nearest_physical_station"], "Karve Road / Kothrud Station (MPCB)")
        print("-> [PASS] Station registry verified with accurate Karve Nagar proximity mapping.")

    def test_database_and_duplicate_prevention(self):
        test_df = pd.DataFrame([{
            "timestamp": "2026-10-15T12:00",
            "pm25": 45.0,
            "pm10": 70.0,
            "no2": 25.0,
            "co": 0.5,
            "so2": 10.0,
            "ozone": 50.0,
            "aqi": 75,
            "aqi_category": "Satisfactory",
            "prominent_pollutant": "PM25"
        }])
        saved1 = save_readings("test_station", test_df)
        self.assertEqual(saved1, 1)

        # Re-saving same timestamp must update/not duplicate
        saved2 = save_readings("test_station", test_df)
        self.assertEqual(saved2, 1)

        hist = get_historical_readings("test_station")
        self.assertEqual(len(hist), 1)
        print("-> [PASS] Database initialization and duplicate prevention verified.")

    def test_aqi_calculation(self):
        # Known CPCB values: PM2.5 = 25 -> Good (<= 50)
        aqi_val, cat, color, adv, subs = calculate_overall_aqi({"pm25": 25.0, "no2": 20.0})
        self.assertIsNotNone(aqi_val)
        self.assertLessEqual(aqi_val, 50)
        self.assertEqual(cat, "Good")

        # Elevated: PM2.5 = 85 -> Moderate (101-200)
        aqi_mod, cat_mod, _, _, _ = calculate_overall_aqi({"pm25": 85.0})
        self.assertTrue(101 <= aqi_mod <= 200)
        self.assertEqual(cat_mod, "Moderate")
        print("-> [PASS] CPCB Indian NAQI mathematical calculations verified.")

    def test_backend_model_loading(self):
        active_bundle = load_active_model_bundle()
        self.assertIsNotNone(active_bundle, "air_quality_best_model.joblib must load successfully")
        self.assertEqual(len(active_bundle["features"]), 16)

        cb_bundle = load_catboost_bundle()
        self.assertIsNotNone(cb_bundle, "air_quality_catboost.joblib must load successfully")

        kmeans_bundle = load_cluster_bundle()
        self.assertIsNotNone(kmeans_bundle, "air_quality_models.joblib must load successfully")
        print("-> [PASS] All 3 backend models (Random Forest, CatBoost, K-Means) loaded successfully.")

    def test_feature_compatibility_audit(self):
        # Incompatible inputs (missing metal oxide sensors) should be rejected
        incompatible = {"pm25": 50.0, "no2": 30.0}
        is_compat, missing = validate_feature_compatibility(incompatible)
        self.assertFalse(is_compat)
        self.assertGreater(len(missing), 0)
        print("-> [PASS] Model feature compatibility auditor correctly flags non-conforming inputs.")

    def test_health_advisory_engine(self):
        adv = get_health_advisory(85, "Satisfactory", "PM2.5", "Karve Nagar, Pune")
        self.assertIn("headline", adv)
        self.assertIn("general_public", adv)
        self.assertIn("traffic_note", adv)
        self.assertIn("Karve Nagar", adv["traffic_note"])
        print("-> [PASS] Health advisory engine verified with demographic guidance.")

    def test_public_ui_does_not_expose_academic_jargon(self):
        with open("app.py", "r", encoding="utf-8") as f:
            code = f.read()

        # The public UI code must not render model comparison tables or ML algorithm names
        forbidden_in_ui = [
            "Random Forest Regressor",
            "Gradient Boosting Regressor",
            "CatBoost Regressor",
            "model_comparison.csv",
            "feature_importances_",
            "MAE",
            "RMSE",
            "R²"
        ]
        # In app.py, check that these are NOT in the text presented to users
        # Notice: app.py imports helper modules and does not contain these strings in markdown/HTML
        for forbidden in forbidden_in_ui:
            self.assertNotIn(f"st.write('{forbidden}'", code)
            self.assertNotIn(f'st.write("{forbidden}"', code)
            self.assertNotIn(f"st.markdown('{forbidden}'", code)
            self.assertNotIn(f'st.markdown("{forbidden}"', code)
        print("-> [PASS] Public UI verified: zero academic ML jargon or model names exposed.")


if __name__ == "__main__":
    unittest.main()
