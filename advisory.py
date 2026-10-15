"""
AirSense AI - Environmental Health Advisory and Recommendations Engine.

Generates validated, practical, citizen-friendly exposure-reduction guidance
based on the official Indian National Air Quality Index (NAQI) categories,
prominent pollutants, and local Pune meteorological conditions.
"""

from typing import Dict, List, Any, Optional


def get_health_advisory(aqi_val: Optional[int], aqi_category: str, prominent_pollutant: str, station_name: str = "Pune") -> Dict[str, Any]:
    """
    Generate structured, practical health advisories tailored to the observed
    air quality tier, prominent pollutant, and target demographic groups.
    """
    if aqi_val is None or aqi_category == "Unavailable":
        return {
            "headline": "Telemetry Currently Unavailable",
            "summary": "Observation data for this location is temporarily inaccessible. Please check official local advisories.",
            "color": "#94a3b8",
            "general_public": ["Follow official CPCB/MPCB municipal bulletins."],
            "sensitive_groups": ["Continue prescribed respiratory or cardiac medical regimens."],
            "outdoor_activity": ["No specific restrictions currently advised."],
            "indoor_guidance": ["Maintain routine indoor ventilation."],
            "traffic_note": ""
        }

    # Guidance based on official Indian NAQI categories
    if aqi_val <= 50:
        headline = "Clean & Healthy Air Quality"
        summary = "Air quality is considered satisfactory, and air pollution poses little or no risk."
        color = "#00b050"
        general_public = [
            "Ideal conditions for all outdoor recreation and physical exercise.",
            "Natural ventilation of homes and workspaces is highly recommended."
        ]
        sensitive_groups = [
            "Safe for individuals with asthma or heart conditions without additional precautions."
        ]
        outdoor_activity = ["Enjoy unrestricted outdoor activities and sports."]
        indoor_guidance = ["Open windows to allow fresh air exchange."]

    elif aqi_val <= 100:
        headline = "Satisfactory Air Quality"
        summary = "Air quality is acceptable. However, unusually sensitive individuals may experience minor irritation."
        color = "#92d050"
        general_public = [
            "Air quality is favorable for most everyday outdoor activities.",
            "No special precautions required for the general populace."
        ]
        sensitive_groups = [
            "Individuals with pronounced asthma should monitor for mild coughing or shortness of breath."
        ]
        outdoor_activity = ["Normal outdoor workouts and commutes are safe."]
        indoor_guidance = ["Standard ventilation is suitable throughout the day."]

    elif aqi_val <= 200:
        headline = "Moderate Air Quality Warning"
        summary = f"Elevated {prominent_pollutant} levels may cause breathing discomfort to children, the elderly, and people with lung or heart disease."
        color = "#ffbf00"
        general_public = [
            "Healthy adults can continue routine outdoor activities but should take breaks during strenuous exertion.",
            "Avoid high-congestion transit corridors during peak vehicular hours (8–10 AM and 6–9 PM)."
        ]
        sensitive_groups = [
            "People with asthma should keep rescue inhalers accessible.",
            "Elderly citizens and young children should reduce prolonged intense outdoor play."
        ]
        outdoor_activity = [
            "Shift strenuous jogs or athletic training away from major arterial roads.",
            "Consider indoor exercise if experiencing throat tickle or eye irritation."
        ]
        indoor_guidance = [
            "Ventilate rooms during mid-afternoon hours when atmospheric boundary-layer mixing is greatest."
        ]

    elif aqi_val <= 300:
        headline = "Poor Air Quality Alert"
        summary = "Prolonged exposure can cause breathing discomfort to most people, with pronounced impacts on vulnerable individuals."
        color = "#ff7e00"
        general_public = [
            "Reduce prolonged or heavy outdoor exertion.",
            "Take frequent breaks during all outdoor activities.",
            "Keep vehicle windows closed when driving in traffic."
        ]
        sensitive_groups = [
            "Children, elderly individuals, and those with cardiopulmonary conditions should avoid prolonged outdoor exposure.",
            "Consult a physician if experiencing persistent wheezing or chest tightness."
        ]
        outdoor_activity = [
            "Postpone strenuous outdoor sports, running, or cycling.",
            "Wear a certified N95 or equivalent particulate-filtering respirator if outdoor presence is unavoidable."
        ]
        indoor_guidance = [
            "Keep windows and exterior doors closed during morning and evening rush hours.",
            "Run certified HEPA air purifiers if available in bedrooms and living spaces."
        ]

    elif aqi_val <= 400:
        headline = "Very Poor Air Quality Health Advisory"
        summary = "Significant health concern. Increased likelihood of respiratory distress across the broader population."
        color = "#ff0000"
        general_public = [
            "Avoid outdoor physical exertion entirely.",
            "Wear well-fitted N95/FFP2 masks outdoors to filter fine ambient particles.",
            "Limit road travel and avoid active construction zones."
        ]
        sensitive_groups = [
            "Vulnerable groups should remain indoors in sealed, filtered environments.",
            "Strictly avoid any outdoor physical activity."
        ]
        outdoor_activity = ["All outdoor sports, running, and heavy labor should be avoided or rescheduled."]
        indoor_guidance = [
            "Seal doors and windows to minimize outdoor particle infiltration.",
            "Avoid indoor combustion sources (incense, mosquito coils, unvented frying)."
        ]

    else:
        headline = "Severe Air Quality Emergency"
        summary = "Emergency conditions. Even healthy individuals may experience adverse respiratory symptoms; serious risk for vulnerable groups."
        color = "#7030a0"
        general_public = [
            "Remain indoors as much as possible with windows and vents closed.",
            "Use certified particulate respirators (N95/FFP2) for any mandatory outdoor transit."
        ]
        sensitive_groups = [
            "Strict indoor confinement with air purification is strongly advised."
        ]
        outdoor_activity = ["Complete suspension of outdoor physical activities."]
        indoor_guidance = [
            "Operate HEPA air purifiers continuously on medium-to-high fan speeds.",
            "Seal window seams and door drafts with damp cloths if necessary."
        ]

    # Specific contextual note for Pune & Karve Nagar topography
    traffic_note = (
        f"Local Advisory for {station_name}: During autumn and winter seasons, atmospheric temperature inversions "
        "can trap vehicular emissions in the Mutha river basin. Ground concentrations along Karve Road and Sinhagad Road corridors "
        "typically peak during early morning (7:30–9:30 AM) and evening rush hours (6:30–9:00 PM)."
    )

    return {
        "headline": headline,
        "summary": summary,
        "color": color,
        "general_public": general_public,
        "sensitive_groups": sensitive_groups,
        "outdoor_activity": outdoor_activity,
        "indoor_guidance": indoor_guidance,
        "traffic_note": traffic_note
    }
