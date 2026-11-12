"""Run utilities and merge for unsupported brief turnovers."""
import numpy as np

def runs(y):
    edges = np.r_[0, np.flatnonzero(y[1:] != y[:-1]) + 1, len(y)]
    return [(a, b, y[a]) for a, b in zip(edges[:-1], edges[1:])]

def merge(pred, f, fps, c):
    out = np.array(pred, copy=True)
    rr = runs(np.asarray(pred))
    for left, middle, right in zip(rr[:-2], rr[1:-1], rr[2:]):
        a, b, team = middle
        if left[2] != right[2] or left[2] not in ('H', 'A') or team not in ('H', 'A') or (team == left[2]) or ((b - a) / fps > c['max_seconds']):
            continue
        distance = f[a:b, 0 if team == 'H' else 1]
        valid = np.isfinite(distance)
        if valid.mean() >= c['min_observed_fraction'] and np.all(distance[valid] > c['opponent_min_distance_m']):
            out[a:b] = left[2]
    return out
