"""
AirSense AI - Official Indian National Air Quality Index (NAQI) Engine.

Implements the official Central Pollution Control Board (CPCB), Ministry of Environment,
Forest and Climate Change (MoEFCC) mathematical breakpoints and sub-index formulas.
"""

from typing import Dict, Optional, Tuple, Any

# CPCB NAQI Breakpoint Table: (C_low, C_high, I_low, I_high)
# Units: PM2.5 (ug/m3), PM10 (ug/m3), NO2 (ug/m3), SO2 (ug/m3), CO (mg/m3), O3 (ug/m3)
BREAKPOINTS = {
    "pm25": [
        (0.0, 30.0, 0, 50),
        (31.0, 60.0, 51, 100),
        (61.0, 90.0, 101, 200),
        (91.0, 120.0, 201, 300),
        (121.0, 250.0, 301, 400),
        (250.1, 500.0, 401, 500),
    ],
    "pm10": [
        (0.0, 50.0, 0, 50),
        (51.0, 100.0, 51, 100),
        (101.0, 250.0, 101, 200),
        (251.0, 350.0, 201, 300),
        (351.0, 430.0, 301, 400),
        (430.1, 600.0, 401, 500),
    ],
    "no2": [
        (0.0, 40.0, 0, 50),
        (41.0, 80.0, 51, 100),
        (81.0, 180.0, 101, 200),
        (181.0, 280.0, 201, 300),
        (281.0, 400.0, 301, 400),
        (400.1, 600.0, 401, 500),
    ],
    "so2": [
        (0.0, 40.0, 0, 50),
        (41.0, 80.0, 51, 100),
        (81.0, 380.0, 101, 200),
        (381.0, 800.0, 201, 300),
        (801.0, 1600.0, 301, 400),
        (1600.1, 2500.0, 401, 500),
    ],
    "co": [
        (0.0, 1.0, 0, 50),
        (1.1, 2.0, 51, 100),
        (2.1, 10.0, 101, 200),
        (10.1, 17.0, 201, 300),
        (17.1, 34.0, 301, 400),
        (34.1, 60.0, 401, 500),
    ],
    "ozone": [
        (0.0, 50.0, 0, 50),
        (51.0, 100.0, 51, 100),
        (101.0, 168.0, 101, 200),
        (169.0, 208.0, 201, 300),
        (209.0, 748.0, 301, 400),
        (748.1, 1000.0, 401, 500),
    ]
}

CATEGORIES = [
    (0, 50, "Good", "#00b050", "Minimal health impact. Clean and safe air."),
    (51, 100, "Satisfactory", "#92d050", "Minor breathing discomfort to sensitive people."),
    (101, 200, "Moderate", "#ffbf00", "Breathing discomfort to sensitive individuals and asthma patients."),
    (201, 300, "Poor", "#ff7e00", "Breathing discomfort to most people on prolonged exposure."),
    (301, 400, "Very Poor", "#ff0000", "Respiratory illness on prolonged exposure; avoid outdoor exertion."),
    (401, 500, "Severe", "#7030a0", "Affects healthy people; serious impacts on vulnerable populations.")
]


def calculate_sub_index(pollutant: str, concentration: float) -> Optional[int]:
    """Calculate single pollutant sub-index based on CPCB piecewise linear formula."""
    if concentration is None or concentration < 0:
        return None

    # CO in API is often in ug/m3, convert to mg/m3 if > 50 (e.g., 239 ug/m3 -> 0.239 mg/m3)
    if pollutant == "co" and concentration > 50:
        concentration = concentration / 1000.0

    table = BREAKPOINTS.get(pollutant.lower())
    if not table:
        return None

    for c_low, c_high, i_low, i_high in table:
        if c_low <= concentration <= c_high:
            sub = i_low + ((i_high - i_low) / (c_high - c_low)) * (concentration - c_low)
            return int(round(sub))

    # Cap if above extreme range
    if concentration > table[-1][1]:
        return 500
    return None


def calculate_overall_aqi(readings: Dict[str, Optional[float]]) -> Tuple[Optional[int], str, str, str, Dict[str, int]]:
    """
    Calculate composite CPCB Indian National AQI.
    Returns: (aqi_value, category_name, color_hex, health_statement, sub_indices_dict)
    """
    sub_indices: Dict[str, int] = {}

    for pol, val in readings.items():
        if val is not None:
            sub = calculate_sub_index(pol, val)
            if sub is not None:
                sub_indices[pol] = sub

    if not sub_indices:
        return None, "Unavailable", "#94a3b8", "Insufficient measurement telemetry.", {}

    overall_aqi = max(sub_indices.values())

    # Find prominent pollutant
    prominent = max(sub_indices, key=sub_indices.get)
    pol_names = {
        "pm25": "PM2.5",
        "pm10": "PM10",
        "no2": "NO₂",
        "so2": "SO₂",
        "co": "CO",
        "ozone": "Ozone (O₃)"
    }
    prominent_label = pol_names.get(prominent, prominent.upper())

    # Match category
    for i_low, i_high, cat_name, color, advisory in CATEGORIES:
        if i_low <= overall_aqi <= i_high:
            return overall_aqi, cat_name, color, advisory, sub_indices

    # Above 500 is Severe
    return overall_aqi, "Severe", "#7030a0", "Emergency air quality advisory.", sub_indices


def get_category_meta(aqi_val: Optional[int]) -> Tuple[str, str, str]:
    """Retrieve category name, color, and summary for an AQI score."""
    if aqi_val is None:
        return "Unavailable", "#94a3b8", "No valid data recorded."

    for i_low, i_high, name, color, summary in CATEGORIES:
        if i_low <= aqi_val <= i_high:
            return name, color, summary

    return "Severe", "#7030a0", "Emergency conditions."
