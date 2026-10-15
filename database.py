"""
AirSense AI - Persistent SQLite Telemetry Database.

Manages relational schema, station metadata, duplicate prevention,
historical observations retention, and synchronization audit logs.
"""

import os
import sqlite3
import datetime
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "airsense.db")


def get_connection():
    """Establish connection to local SQLite database with row factory enabled."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize database tables, unique constraints, and seed stations registry."""
    from data.stations import MAHARASHTRA_STATIONS, PUNE_STATIONS

    conn = get_connection()
    cursor = conn.cursor()

    # 1. Stations Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS stations (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        locality TEXT,
        city TEXT,
        state TEXT,
        latitude REAL,
        longitude REAL,
        is_physical_station INTEGER,
        nearest_physical_station TEXT,
        distance_to_physical_km REAL,
        telemetry_type TEXT,
        provider TEXT,
        attribution TEXT,
        last_sync_time TEXT
    )
    """)

    # 2. Hourly Observations Table with Unique Constraint for Duplicate Prevention
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS readings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        station_id TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        pm25 REAL,
        pm10 REAL,
        no2 REAL,
        co REAL,
        so2 REAL,
        ozone REAL,
        aqi INTEGER,
        aqi_category TEXT,
        prominent_pollutant TEXT,
        source TEXT,
        created_at TEXT,
        UNIQUE(station_id, timestamp)
    )
    """)

    # 3. Create indices for fast time-series queries
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_readings_station_time ON readings(station_id, timestamp)")

    # 4. Sync Audit Log
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS sync_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        station_id TEXT,
        status TEXT,
        records_count INTEGER,
        error_message TEXT,
        synced_at TEXT
    )
    """)

    # Seed or update stations from registry
    for s_id, s_data in MAHARASHTRA_STATIONS.items():
        cursor.execute("""
        INSERT INTO stations (
            id, name, locality, city, state, latitude, longitude,
            is_physical_station, nearest_physical_station, distance_to_physical_km,
            telemetry_type, provider, attribution
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            name = excluded.name,
            nearest_physical_station = excluded.nearest_physical_station,
            distance_to_physical_km = excluded.distance_to_physical_km,
            telemetry_type = excluded.telemetry_type,
            provider = excluded.provider,
            attribution = excluded.attribution
        """, (
            s_id, s_data["name"], s_data["locality"], s_data["city"], s_data["state"],
            s_data["latitude"], s_data["longitude"], 1 if s_data["is_physical_station"] else 0,
            s_data["nearest_physical_station"], s_data["distance_to_physical_km"],
            s_data["telemetry_type"], s_data["provider"], s_data["attribution"]
        ))

    conn.commit()
    conn.close()


def save_readings(station_id: str, readings_df: pd.DataFrame, source_label: str = "Open-Meteo/CAMS") -> int:
    """
    Persist DataFrame of hourly readings into SQLite.
    Uses INSERT OR REPLACE to update existing hourly records without creating duplicates.
    Returns the count of saved records.
    """
    if readings_df is None or readings_df.empty:
        return 0

    now_iso = datetime.datetime.now().isoformat()
    conn = get_connection()
    cursor = conn.cursor()
    saved_count = 0

    for _, row in readings_df.iterrows():
        ts = str(row.get("timestamp"))
        if not ts or ts == "nan":
            continue

        cursor.execute("""
        INSERT INTO readings (
            station_id, timestamp, pm25, pm10, no2, co, so2, ozone,
            aqi, aqi_category, prominent_pollutant, source, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(station_id, timestamp) DO UPDATE SET
            pm25 = COALESCE(excluded.pm25, readings.pm25),
            pm10 = COALESCE(excluded.pm10, readings.pm10),
            no2 = COALESCE(excluded.no2, readings.no2),
            co = COALESCE(excluded.co, readings.co),
            so2 = COALESCE(excluded.so2, readings.so2),
            ozone = COALESCE(excluded.ozone, readings.ozone),
            aqi = COALESCE(excluded.aqi, readings.aqi),
            aqi_category = COALESCE(excluded.aqi_category, readings.aqi_category),
            prominent_pollutant = COALESCE(excluded.prominent_pollutant, readings.prominent_pollutant),
            created_at = excluded.created_at
        """, (
            station_id,
            ts,
            row.get("pm25"),
            row.get("pm10"),
            row.get("no2"),
            row.get("co"),
            row.get("so2"),
            row.get("ozone"),
            int(row["aqi"]) if pd.notna(row.get("aqi")) else None,
            row.get("aqi_category"),
            row.get("prominent_pollutant"),
            source_label,
            now_iso
        ))
        saved_count += 1

    # Update station's last sync time
    cursor.execute("""
    UPDATE stations SET last_sync_time = ? WHERE id = ?
    """, (now_iso, station_id))

    conn.commit()
    conn.close()
    return saved_count


def get_latest_reading(station_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve the most recent observation for a given station."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT * FROM readings
    WHERE station_id = ?
    ORDER BY timestamp DESC
    LIMIT 1
    """, (station_id,))
    row = cursor.fetchone()
    conn.close()

    if row:
        return dict(row)
    return None


def get_historical_readings(
    station_id: str,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    limit: int = 3000
) -> pd.DataFrame:
    """Retrieve historical observations for a station filtered by timestamp window."""
    conn = get_connection()
    query = "SELECT * FROM readings WHERE station_id = ?"
    params: List[Any] = [station_id]

    if start_time:
        query += " AND timestamp >= ?"
        params.append(start_time)
    if end_time:
        query += " AND timestamp <= ?"
        params.append(end_time)

    query += " ORDER BY timestamp ASC LIMIT ?"
    params.append(limit)

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    if not df.empty and "timestamp" in df.columns:
        df["DateTime"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def get_station_metadata(station_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve station description, coordinates, and physical proximity details."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM stations WHERE id = ?", (station_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None


def get_last_sync_info(station_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve the latest sync audit entry for a station."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT * FROM sync_log
    WHERE station_id = ?
    ORDER BY synced_at DESC
    LIMIT 1
    """, (station_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None


def log_sync(station_id: str, status: str, count: int, error_message: str = ""):
    """Log an ingestion or refresh attempt into sync_log."""
    now_iso = datetime.datetime.now().isoformat()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO sync_log (station_id, status, records_count, error_message, synced_at)
    VALUES (?, ?, ?, ?, ?)
    """, (station_id, status, count, error_message, now_iso))
    conn.commit()
    conn.close()
