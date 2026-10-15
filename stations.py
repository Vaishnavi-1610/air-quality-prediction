"""

AirSense AI - Verified Station and Locality Registry for Maharashtra Cities.



Provides accurate geographical coordinates, monitoring station designations,

and physical station proximity mapping for cities across Maharashtra:

Pune, Mumbai, Nagpur, Nashik, Thane, Chhatrapati Sambhajinagar, Kolhapur, Solapur, Amravati.

"""



from typing import Dict, List, Any



MAHARASHTRA_STATIONS: Dict[str, Dict[str, Any]] = {

    # -------------------------------------------------------------------------

    # 1. PUNE

    # -------------------------------------------------------------------------

    "pune_karve_nagar": {

        "id": "pune_karve_nagar",

        "name": "Karve Nagar, Pune",

        "locality": "Karve Nagar",

        "city": "Pune",

        "state": "Maharashtra",

        "latitude": 18.4914,

        "longitude": 73.8217,

        "is_physical_station": False,

        "locality_description": (

            "Prominent residential and educational hub in South-West Pune. "

            "Does not have an on-site CPCB CAAQMS physical tower; monitored via high-resolution "

            "atmospheric grid telemetry paired with reference from nearby Karve Road station."

        ),

        "nearest_physical_station": "Karve Road / Kothrud Station (MPCB)",

        "distance_to_physical_km": 2.4,

        "telemetry_type": "High-Resolution Gridded Atmospheric Telemetry (CAMS) + Nearby CAAQMS Reference",

        "provider": "Copernicus Atmosphere Monitoring Service / Open-Meteo & MPCB",

        "attribution": "Atmospheric data provided via Copernicus CAMS & MPCB Pune Network."

    },

    "pune_karve_road": {

        "id": "pune_karve_road",

        "name": "Karve Road / Kothrud Station, Pune",

        "locality": "Kothrud (Karve Road)",

        "city": "Pune",

        "state": "Maharashtra",

        "latitude": 18.5028,

        "longitude": 73.8184,

        "is_physical_station": True,

        "locality_description": "Major urban transit corridor in West Pune. Continuous ground monitoring station.",

        "nearest_physical_station": "Karve Road / Kothrud Station (MPCB)",

        "distance_to_physical_km": 0.0,

        "telemetry_type": "Physical Continuous Ambient Air Quality Monitoring Station (CAAQMS)",

        "provider": "Maharashtra Pollution Control Board (MPCB) / CPCB",

        "attribution": "Official ground station operated under National Ambient Air Quality Monitoring Network."

    },

    "pune_shivajinagar": {

        "id": "pune_shivajinagar",

        "name": "Shivajinagar Station, Pune",

        "locality": "Shivajinagar",

        "city": "Pune",

        "state": "Maharashtra",

        "latitude": 18.5314,

        "longitude": 73.8446,

        "is_physical_station": True,

        "locality_description": "Central commercial and transport junction in Pune. Primary reference station.",

        "nearest_physical_station": "Shivajinagar Station",

        "distance_to_physical_km": 0.0,

        "telemetry_type": "Physical Continuous Ambient Air Quality Monitoring Station (CAAQMS)",

        "provider": "SAFAR / Indian Institute of Tropical Meteorology (IITM) & MPCB",

        "attribution": "Continuous station operated by SAFAR-India & MPCB."

    },

    "pune_pashan": {

        "id": "pune_pashan",

        "name": "Pashan (IITM Campus) Station, Pune",

        "locality": "Pashan",

        "city": "Pune",

        "state": "Maharashtra",

        "latitude": 18.5362,

        "longitude": 73.8055,

        "is_physical_station": True,

        "locality_description": "Institutional green zone in West Pune, housing IITM atmospheric research campus.",

        "nearest_physical_station": "Pashan Station",

        "distance_to_physical_km": 0.0,

        "telemetry_type": "Physical Continuous Ambient Air Quality Monitoring Station (CAAQMS)",

        "provider": "SAFAR-IITM Pune",

        "attribution": "Atmospheric research station operated by Ministry of Earth Sciences."

    },

    "pune_hadapsar": {

        "id": "pune_hadapsar",

        "name": "Hadapsar Station, Pune",

        "locality": "Hadapsar",

        "city": "Pune",

        "state": "Maharashtra",

        "latitude": 18.5018,

        "longitude": 73.9260,

        "is_physical_station": True,

        "locality_description": "Major commercial, IT, and industrial suburb in East Pune.",

        "nearest_physical_station": "Hadapsar Station",

        "distance_to_physical_km": 0.0,

        "telemetry_type": "Physical Continuous Ambient Air Quality Monitoring Station (CAAQMS)",

        "provider": "SAFAR / MPCB Pune",

        "attribution": "Continuous monitoring network under SAFAR / MPCB."

    },

    "pune_katraj": {

        "id": "pune_katraj",

        "name": "Katraj Station, Pune",

        "locality": "Katraj",

        "city": "Pune",

        "state": "Maharashtra",

        "latitude": 18.4575,

        "longitude": 73.8587,

        "is_physical_station": True,

        "locality_description": "South Pune gateway and suburban educational hub.",

        "nearest_physical_station": "Katraj Station",

        "distance_to_physical_km": 0.0,

        "telemetry_type": "Physical Continuous Ambient Air Quality Monitoring Station (CAAQMS)",

        "provider": "MPCB Pune",

        "attribution": "Air quality monitoring station operated by MPCB."

    },



    # -------------------------------------------------------------------------

    # 2. MUMBAI

    # -------------------------------------------------------------------------

    "mumbai_bkc": {

        "id": "mumbai_bkc",

        "name": "Bandra Kurla Complex (BKC), Mumbai",

        "locality": "Bandra Kurla Complex",

        "city": "Mumbai",

        "state": "Maharashtra",

        "latitude": 19.0660,

        "longitude": 72.8687,

        "is_physical_station": True,

        "locality_description": "Central business district in Mumbai with high traffic movement.",

        "nearest_physical_station": "BKC Station (MPCB)",

        "distance_to_physical_km": 0.0,

        "telemetry_type": "Physical Continuous Ambient Air Quality Monitoring Station (CAAQMS)",

        "provider": "Maharashtra Pollution Control Board (MPCB)",

        "attribution": "Official ground station operated under CPCB/MPCB CAAQMS Network."

    },

    "mumbai_colaba": {

        "id": "mumbai_colaba",

        "name": "Colaba / South Mumbai Station",

        "locality": "Colaba",

        "city": "Mumbai",

        "state": "Maharashtra",

        "latitude": 18.9067,

        "longitude": 72.8147,

        "is_physical_station": True,

        "locality_description": "Coastal peninsula in South Mumbai with maritime breeze influences.",

        "nearest_physical_station": "Colaba Station",

        "distance_to_physical_km": 0.0,

        "telemetry_type": "Physical Continuous Ambient Air Quality Monitoring Station (CAAQMS)",

        "provider": "MPCB / SAFAR Mumbai",

        "attribution": "Coastal monitoring station operated by SAFAR / MPCB."

    },

    "mumbai_worli": {

        "id": "mumbai_worli",

        "name": "Worli Station, Mumbai",

        "locality": "Worli",

        "city": "Mumbai",

        "state": "Maharashtra",

        "latitude": 19.0166,

        "longitude": 72.8170,

        "is_physical_station": True,

        "locality_description": "Central waterfront commercial and residential district in Mumbai.",

        "nearest_physical_station": "Worli Station",

        "distance_to_physical_km": 0.0,

        "telemetry_type": "Physical Continuous Ambient Air Quality Monitoring Station (CAAQMS)",

        "provider": "MPCB Mumbai",

        "attribution": "Continuous station operated by MPCB."

    },



    # -------------------------------------------------------------------------

    # 3. NAGPUR

    # -------------------------------------------------------------------------

    "nagpur_civil_lines": {

        "id": "nagpur_civil_lines",

        "name": "Civil Lines Station, Nagpur",

        "locality": "Civil Lines",

        "city": "Nagpur",

        "state": "Maharashtra",

        "latitude": 21.1557,

        "longitude": 79.0763,

        "is_physical_station": True,

        "locality_description": "Administrative and institutional center of Nagpur, eastern Maharashtra.",

        "nearest_physical_station": "Civil Lines Station",

        "distance_to_physical_km": 0.0,

        "telemetry_type": "Physical Continuous Ambient Air Quality Monitoring Station (CAAQMS)",

        "provider": "MPCB Nagpur",

        "attribution": "Continuous monitoring network operated by MPCB."

    },



    # -------------------------------------------------------------------------

    # 4. NASHIK

    # -------------------------------------------------------------------------

    "nashik_gangapur": {

        "id": "nashik_gangapur",

        "name": "Gangapur Road Station, Nashik",

        "locality": "Gangapur Road",

        "city": "Nashik",

        "state": "Maharashtra",

        "latitude": 20.0125,

        "longitude": 73.7650,

        "is_physical_station": True,

        "locality_description": "Prominent urban and educational corridor in Nashik, northern Maharashtra.",

        "nearest_physical_station": "Gangapur Road Station",

        "distance_to_physical_km": 0.0,

        "telemetry_type": "Physical Continuous Ambient Air Quality Monitoring Station (CAAQMS)",

        "provider": "MPCB Nashik",

        "attribution": "Air monitoring station operated by MPCB."

    },



    # -------------------------------------------------------------------------

    # 5. THANE

    # -------------------------------------------------------------------------

    "thane_naupada": {

        "id": "thane_naupada",

        "name": "Naupada Station, Thane",

        "locality": "Naupada",

        "city": "Thane",

        "state": "Maharashtra",

        "latitude": 19.1932,

        "longitude": 72.9712,

        "is_physical_station": True,

        "locality_description": "Dense commercial and transit hub in central Thane.",

        "nearest_physical_station": "Naupada Station",

        "distance_to_physical_km": 0.0,

        "telemetry_type": "Physical Continuous Ambient Air Quality Monitoring Station (CAAQMS)",

        "provider": "MPCB / Thane Municipal Corporation",

        "attribution": "Continuous station operated under MPCB CAAQMS Network."

    },



    # -------------------------------------------------------------------------

    # 6. CHHATRAPATI SAMBHAJINAGAR (AURANGABAD)

    # -------------------------------------------------------------------------

    "csn_cidco": {

        "id": "csn_cidco",

        "name": "CIDCO Station, Chhatrapati Sambhajinagar",

        "locality": "CIDCO",

        "city": "Chhatrapati Sambhajinagar",

        "state": "Maharashtra",

        "latitude": 19.8825,

        "longitude": 75.3570,

        "is_physical_station": True,

        "locality_description": "Major commercial and residential suburb in Chhatrapati Sambhajinagar (Marathwada).",

        "nearest_physical_station": "CIDCO Station",

        "distance_to_physical_km": 0.0,

        "telemetry_type": "Physical Continuous Ambient Air Quality Monitoring Station (CAAQMS)",

        "provider": "MPCB Sambhajinagar",

        "attribution": "Station operated under MPCB National Ambient Air Quality Network."

    },



    # -------------------------------------------------------------------------

    # 7. KOLHAPUR

    # -------------------------------------------------------------------------

    "kolhapur_university": {

        "id": "kolhapur_university",

        "name": "Shivaji University Station, Kolhapur",

        "locality": "Shivaji University Campus",

        "city": "Kolhapur",

        "state": "Maharashtra",

        "latitude": 16.6775,

        "longitude": 74.2530,

        "is_physical_station": True,

        "locality_description": "Green institutional campus in southern Maharashtra.",

        "nearest_physical_station": "Shivaji University Station",

        "distance_to_physical_km": 0.0,

        "telemetry_type": "Physical Continuous Ambient Air Quality Monitoring Station (CAAQMS)",

        "provider": "MPCB Kolhapur",

        "attribution": "Continuous station operated by MPCB."

    },



    # -------------------------------------------------------------------------

    # 8. SOLAPUR

    # -------------------------------------------------------------------------

    "solapur_saat_rasta": {

        "id": "solapur_saat_rasta",

        "name": "Saat Rasta Station, Solapur",

        "locality": "Saat Rasta",

        "city": "Solapur",

        "state": "Maharashtra",

        "latitude": 17.6698,

        "longitude": 75.9080,

        "is_physical_station": True,

        "locality_description": "Key central junction in Solapur, south-eastern Maharashtra.",

        "nearest_physical_station": "Saat Rasta Station",

        "distance_to_physical_km": 0.0,

        "telemetry_type": "Physical Continuous Ambient Air Quality Monitoring Station (CAAQMS)",

        "provider": "MPCB Solapur",

        "attribution": "Station operated by MPCB."

    },



    # -------------------------------------------------------------------------

    # 9. AMRAVATI

    # -------------------------------------------------------------------------

    "amravati_central": {

        "id": "amravati_central",

        "name": "Rajkamal Square Station, Amravati",

        "locality": "Rajkamal Square",

        "city": "Amravati",

        "state": "Maharashtra",

        "latitude": 20.9320,

        "longitude": 77.7580,

        "is_physical_station": True,

        "locality_description": "Commercial heart of Amravati in the Vidarbha region.",

        "nearest_physical_station": "Rajkamal Square Station",

        "distance_to_physical_km": 0.0,

        "telemetry_type": "Physical Continuous Ambient Air Quality Monitoring Station (CAAQMS)",

        "provider": "MPCB Amravati",

        "attribution": "Air monitoring station operated by MPCB."

    }

}





def get_all_stations() -> Dict[str, Dict[str, Any]]:

    """Return dictionary of all available stations across Maharashtra."""

    return MAHARASHTRA_STATIONS





def get_available_cities() -> List[str]:

    """Return list of supported Maharashtra cities with verified data sources."""

    cities = []

    for s in MAHARASHTRA_STATIONS.values():

        if s["city"] not in cities:

            cities.append(s["city"])

    return cities





def get_stations_for_city(city_name: str) -> Dict[str, Dict[str, Any]]:

    """Retrieve all stations and localities belonging to a specific city."""

    return {k: v for k, v in MAHARASHTRA_STATIONS.items() if v["city"].lower() == city_name.lower()}





def get_station_by_id(station_id: str) -> Dict[str, Any]:

    """Retrieve station metadata by unique ID, defaulting to Karve Nagar, Pune."""

    return MAHARASHTRA_STATIONS.get(station_id, MAHARASHTRA_STATIONS["pune_karve_nagar"])





def get_station_options_for_city(city_name: str) -> Dict[str, str]:

    """Return mapping of display name to station ID for a selected city."""

    matching = get_stations_for_city(city_name)

    return {v["name"]: k for k, v in matching.items()}



def get_station_options() -> Dict[str, str]:
    """Return mapping of every station display name to its unique station ID."""
    return {station["name"]: station_id for station_id, station in MAHARASHTRA_STATIONS.items()}


# Backward compatibility alias for legacy imports
PUNE_STATIONS = MAHARASHTRA_STATIONS

# Backward compatibility aliases for legacy ID lookups
MAHARASHTRA_STATIONS["karve_nagar"] = MAHARASHTRA_STATIONS["pune_karve_nagar"]
MAHARASHTRA_STATIONS["shivajinagar"] = MAHARASHTRA_STATIONS["pune_shivajinagar"]


