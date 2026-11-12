"""Distances, activity and pitch features."""
import numpy as np
from .tracking import Observations

def features(o: Observations):
    d = np.linalg.norm(o.players - o.ball[:, None, :], axis=2)
    nearest = []
    for team in (0, 1):
        a = np.where(np.isfinite(d[:, o.teams == team]), d[:, o.teams == team], np.inf)
        v = a.min(axis=1)
        v[~np.isfinite(v)] = np.nan
        nearest.append(v)
    valid = np.isfinite(o.speeds) & (o.speeds >= 0)
    mean_speed = np.divide(np.where(valid, o.speeds, 0).sum(axis=1), valid.sum(axis=1), out=np.full(len(d), np.nan), where=valid.sum(axis=1) > 0)
    speed = np.full(len(d), np.nan)
    speed[1:] = np.linalg.norm(np.diff(o.ball, axis=0), axis=1) * o.fps
    x, y = o.ball.T
    length, width = o.pitch
    edge = np.minimum.reduce([x, length - x, y, width - y])
    return np.column_stack([*nearest, np.log1p(mean_speed), np.log1p(speed), edge])

def append_contrast(f):
    f = np.asarray(f, dtype=float)
    if f.ndim != 2 or f.shape[1] != 5:
        raise ValueError('Expected original five features')
    good = np.isfinite(f[:, :2]).all(axis=1)
    advantage = np.full(len(f), np.nan)
    advantage[good] = f[good, 1] - f[good, 0]
    return np.column_stack([f, advantage])
