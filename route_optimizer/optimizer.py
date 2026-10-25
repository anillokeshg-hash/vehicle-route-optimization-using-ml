"""Route optimization: clustering + nearest neighbour + 2-opt."""
import numpy as np
from sklearn.cluster import KMeans


def route_cost(route, matrix):
    return sum(matrix[route[i]][route[i + 1]] for i in range(len(route) - 1))


def nearest_neighbour(stop_ids, matrix, depot=0):
    """Greedy route that always visits the closest unvisited stop next."""
    remaining = set(stop_ids)
    route, current = [depot], depot
    while remaining:
        nxt = min(remaining, key=lambda s: matrix[current][s])
        route.append(nxt)
        remaining.remove(nxt)
        current = nxt
    route.append(depot)
    return route


def two_opt(route, matrix):
    """Improve a route by reversing segments while it reduces the cost."""
    best = route[:]
    improved = True
    while improved:
        improved = False
        for i in range(1, len(best) - 2):
            for j in range(i + 1, len(best) - 1):
                candidate = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                if route_cost(candidate, matrix) < route_cost(best, matrix) - 1e-9:
                    best = candidate
                    improved = True
    return best


def assign_to_vehicles(stops, k):
    """Group stops (excluding depot) into k clusters using K-Means."""
    ids = [s["id"] for s in stops if s["id"] != 0]
    if k <= 1 or len(ids) <= k:
        return [ids] if k <= 1 else [[i] for i in ids]
    coords = np.array([[stops[i]["lat"], stops[i]["lon"]] for i in ids])
    labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(coords)
    return [[ids[i] for i in range(len(ids)) if labels[i] == c] for c in range(k)]


def optimize(stops, time_matrix, dist_matrix, vehicles):
    """Return per-vehicle naive vs optimized routes and totals."""
    groups = assign_to_vehicles(stops, vehicles)
    result = []
    for v, group in enumerate(groups):
        if not group:
            continue
        naive = [0] + sorted(group) + [0]  # visit in given order
        best = two_opt(nearest_neighbour(group, time_matrix), time_matrix)
        result.append({
            "vehicle": v + 1,
            "naive_route": naive,
            "route": best,
            "naive_min": round(route_cost(naive, time_matrix), 1),
            "optimized_min": round(route_cost(best, time_matrix), 1),
            "naive_km": round(route_cost(naive, dist_matrix), 1),
            "optimized_km": round(route_cost(best, dist_matrix), 1),
        })
    return result
