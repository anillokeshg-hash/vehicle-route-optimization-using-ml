"""Vehicle Route Optimization Using ML - Flask server."""
import os

from flask import Flask, jsonify, request, send_from_directory

from data import generate_stops
from model import build_matrices, train_model
from optimizer import optimize

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)

print("Training travel-time prediction model...")
model = train_model()


@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/api/optimize")
def api_optimize():
    n = min(max(request.args.get("stops", 10, type=int), 2), 40)
    vehicles = min(max(request.args.get("vehicles", 2, type=int), 1), 6)
    hour = min(max(request.args.get("hour", 9, type=int), 0), 23)
    weekend = 1 if request.args.get("weekend", 0, type=int) else 0
    seed = request.args.get("seed", type=int)

    stops = generate_stops(n, seed)
    dist, time = build_matrices(model, stops, hour, weekend)
    routes = optimize(stops, time, dist, vehicles)

    naive_total = round(sum(r["naive_min"] for r in routes), 1)
    opt_total = round(sum(r["optimized_min"] for r in routes), 1)
    naive_km = round(sum(r["naive_km"] for r in routes), 1)
    opt_km = round(sum(r["optimized_km"] for r in routes), 1)
    saving = round(100 * (naive_total - opt_total) / naive_total, 1) if naive_total else 0

    return jsonify({
        "stops": stops,
        "routes": routes,
        "summary": {
            "naive_min": naive_total, "optimized_min": opt_total,
            "naive_km": naive_km, "optimized_km": opt_km,
            "time_saved_pct": saving,
        },
    })


if __name__ == "__main__":
    app.run(debug=False, port=5000)
