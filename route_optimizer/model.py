"""ML model that predicts travel time between two locations."""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from data import ROAD_FACTOR, generate_trip_data, haversine

FEATURES = ["distance_km", "hour", "weekend"]


def train_model(verbose=True):
    df = generate_trip_data()
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df["travel_min"], test_size=0.2, random_state=7
    )
    model = GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=7)
    model.fit(X_train, y_train)
    if verbose:
        pred = model.predict(X_test)
        print(f"Travel-time model: R2={r2_score(y_test, pred):.3f}, "
              f"MAE={mean_absolute_error(y_test, pred):.2f} min")
    return model


def build_matrices(model, stops, hour, weekend):
    """Return (distance_km matrix, predicted travel-time matrix in minutes)."""
    n = len(stops)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dist[i][j] = ROAD_FACTOR * haversine(
                    stops[i]["lat"], stops[i]["lon"], stops[j]["lat"], stops[j]["lon"]
                )
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    X = pd.DataFrame(
        [[dist[i][j], hour, weekend] for i, j in pairs], columns=FEATURES
    )
    preds = model.predict(X)
    time = np.zeros((n, n))
    for (i, j), p in zip(pairs, preds):
        time[i][j] = max(float(p), 0.5)
    return dist, time


if __name__ == "__main__":
    train_model()
