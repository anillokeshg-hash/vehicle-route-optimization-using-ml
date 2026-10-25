"""Locations and simulated trip data for route optimization."""
import math

import numpy as np
import pandas as pd

# Map centre / depot location (change to your city)
CENTER_LAT, CENTER_LON = 28.6139, 77.2090
ROAD_FACTOR = 1.3  # road distance is ~1.3x straight-line distance


def haversine(lat1, lon1, lat2, lon2):
    """Straight-line distance in km between two coordinates."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def generate_stops(n, seed=None):
    """Depot (index 0) followed by n random delivery stops."""
    rng = np.random.default_rng(seed)
    stops = [{"id": 0, "name": "Depot", "lat": CENTER_LAT, "lon": CENTER_LON}]
    for i in range(1, n + 1):
        stops.append({
            "id": i,
            "name": f"Stop {i}",
            "lat": round(CENTER_LAT + float(rng.uniform(-0.12, 0.12)), 5),
            "lon": round(CENTER_LON + float(rng.uniform(-0.12, 0.12)), 5),
        })
    return stops


def traffic_factor(hour, weekend):
    """How much slower than free-flow traffic is at a given hour."""
    if weekend:
        return 1.15 if 11 <= hour <= 20 else 1.0
    if 8 <= hour <= 10 or 17 <= hour <= 20:
        return 1.8   # rush hour
    if 11 <= hour <= 16:
        return 1.3
    return 1.0


def generate_trip_data(n=6000, seed=7):
    """Simulated historical trips: distance, hour, weekend -> travel time (min)."""
    rng = np.random.default_rng(seed)
    dist = rng.uniform(0.3, 30, n)
    hour = rng.integers(0, 24, n)
    weekend = rng.integers(0, 2, n)
    free_flow_speed = 45.0  # km/h
    minutes = np.array([
        d / free_flow_speed * 60 * traffic_factor(h, w) for d, h, w in zip(dist, hour, weekend)
    ])
    minutes += 2 + rng.normal(0, 1.5, n)  # stop/signal delay + noise
    return pd.DataFrame({
        "distance_km": dist, "hour": hour, "weekend": weekend,
        "travel_min": np.clip(minutes, 1, None),
    })
