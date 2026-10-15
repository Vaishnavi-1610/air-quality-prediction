"""
AirSense AI - Scheduled Data Refresh Script.

Executes standalone synchronization for all registered monitoring stations in Pune,
fetching new hourly observations, computing Indian NAQI, and updating SQLite database.
Suitable for execution via Windows Task Scheduler or automated cron.
"""

import sys
import datetime
from data.database import init_db
from data.stations import get_all_stations
from data.api_client import fetch_and_sync_station_data


def run_refresh(past_days: int = 7):
    """Sync all registered Pune monitoring stations."""
    print("=" * 60)
    print(f"AirSense AI - Automated Telemetry Refresh [{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
    print("=" * 60)

    init_db()
    stations = get_all_stations()
    total_saved = 0
    success_count = 0

    for s_id, s_info in stations.items():
        print(f"\n-> Refreshing station: {s_info['name']} ({s_id})...")
        success, msg, count = fetch_and_sync_station_data(s_id, past_days=past_days)
        if success:
            print(f"   [SUCCESS] {msg}")
            total_saved += count
            success_count += 1
        else:
            print(f"   [WARNING/ERROR] {msg}")

    print("\n" + "=" * 60)
    print(f"Refresh completed: {success_count}/{len(stations)} stations synced.")
    print(f"Total observations updated or verified: {total_saved}")
    print("=" * 60)
    return success_count > 0


if __name__ == "__main__":
    days = 7
    if len(sys.argv) > 1:
        try:
            days = int(sys.argv[1])
        except ValueError:
            pass

    success = run_refresh(past_days=days)
    sys.exit(0 if success else 1)
