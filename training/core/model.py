"""Gaussian observation features and symmetric HMM estimation on external matches."""
import numpy as np
from dataclasses import replace
from dfl import Observations
FEATURES = ['nearest_H_m', 'nearest_A_m', 'log1p_mean_player_speed_mps', 'log1p_ball_speed_mps', 'signed_ball_boundary_distance_m']
STRIDE = 5

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

def degrade(o, seed, missing_fraction=0.4, mean_gap_seconds=5.0):
    """Synthetic Markov ball dropout only; labels and player streams stay unchanged.
    Sensitivity scenario, not a verified simulator of the challenge's occlusion process.
    """
    rng = np.random.default_rng(seed)
    hidden = np.zeros(len(o.ball), bool)
    p_exit = 1 / (mean_gap_seconds * o.fps)
    p_enter = missing_fraction / (1 - missing_fraction) * p_exit
    state = bool(rng.random() < missing_fraction)
    for i, u in enumerate(rng.random(len(hidden))):
        if state:
            state = not u < p_exit
        else:
            state = u < p_enter
        hidden[i] = state
    ball = o.ball.copy()
    ball[hidden] = np.nan
    return replace(o, ball=ball)

def fit(sequences):
    samples = [[] for _ in range(3)]
    transitions = np.ones((3, 3), float)
    initial = np.ones(3, float)
    for f, y in sequences:
        f = f[::STRIDE]
        y = y[::STRIDE]
        codes = np.array([{'H': 0, 'A': 1, 'D': 2, 'X': -1}[v] for v in y])
        for swapped in (False, True):
            ff = f[:, [1, 0, 2, 3, 4]] if swapped else f
            yy = np.where(codes == 0, 1, np.where(codes == 1, 0, codes)) if swapped else codes
            for c in range(3):
                samples[c].append(ff[yy == c])
            if yy[0] >= 0:
                initial[yy[0]] += 1
            mask = (yy[:-1] >= 0) & (yy[1:] >= 0)
            np.add.at(transitions, (yy[:-1][mask], yy[1:][mask]), 1)
    means = []
    variances = []
    counts = []
    for chunks in samples:
        x = np.concatenate(chunks)
        count = np.isfinite(x).sum(axis=0)
        if np.any(count < 10):
            raise ValueError('Insufficient external training evidence')
        means.append(np.nanmean(x, axis=0))
        variances.append(np.maximum(np.nanvar(x, axis=0), 1e-06))
        counts.append(count.tolist())
    return dict(schema=1, feature_names=FEATURES, stride=STRIDE, fps=25, means=np.asarray(means).tolist(), variances=np.asarray(variances).tolist(), transitions=(transitions / transitions.sum(axis=1, keepdims=True)).tolist(), initial=(initial / initial.sum()).tolist(), training_feature_counts=counts)

def predict(f, params):
    stride = params['stride']
    x = f[::stride]
    mu = np.asarray(params['means'])
    var = np.asarray(params['variances'])
    contributions = -0.5 * ((x[:, None, :] - mu) ** 2 / var + np.log(2 * np.pi * var))
    emission = np.where(np.isfinite(x[:, None, :]), contributions, 0).sum(axis=2)
    if not np.isfinite(emission).all():
        raise ValueError('Nonfinite model emissions')
    trans = np.log(params['transitions'])
    delta = np.log(params['initial']) + emission[0]
    back = np.zeros((len(x), 3), np.int8)
    for t in range(1, len(x)):
        candidate = delta[:, None] + trans
        back[t] = candidate.argmax(axis=0)
        delta = candidate.max(axis=0) + emission[t]
    path = np.zeros(len(x), np.int8)
    path[-1] = delta.argmax()
    for t in range(len(x) - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return np.repeat(np.array(['H', 'A', 'D'])[path], stride)[:len(f)]
