"""Gaussian evidence, odds fusion and Viterbi decoding."""
import numpy as np

def gaussian_emissions(x, means, variances):
    x = np.asarray(x, dtype=float)
    mu = np.asarray(means)
    var = np.asarray(variances)
    terms = -0.5 * ((x[:, None, :] - mu) ** 2 / var + np.log(2 * np.pi * var))
    result = np.where(np.isfinite(x[:, None, :]), terms, 0).sum(axis=2)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite emissions')
    return result

def fuse(base, motion_odds):
    base = np.asarray(base, dtype=float)
    delta = np.asarray(motion_odds, dtype=float)
    if base.ndim != 2 or base.shape[1] != 3 or delta.shape != (len(base),):
        raise ValueError('Invalid fusion shape')
    if not np.isfinite(base).all() or not np.isfinite(delta).all():
        raise ValueError('Nonfinite fusion input')
    out = base.copy()
    use = delta != 0
    alive = np.logaddexp(base[use, 0], base[use, 1])
    ratio = base[use, 0] - base[use, 1] + delta[use]
    out[use, 0] = alive - np.logaddexp(0, -ratio)
    out[use, 1] = alive - np.logaddexp(0, ratio)
    if not np.isfinite(out).all():
        raise ValueError('Nonfinite fused emissions')
    return out

def decode(emission, params, n_frames):
    """Same Viterbi recurrence and expansion as model.predict, accepting emissions."""
    emission = np.asarray(emission)
    if len(emission) == 0 or not np.isfinite(emission).all():
        raise ValueError('Invalid decode emissions')
    trans = np.log(params['transitions'])
    delta = np.log(params['initial']) + emission[0]
    back = np.zeros((len(emission), 3), np.int8)
    for t in range(1, len(emission)):
        candidate = delta[:, None] + trans
        back[t] = candidate.argmax(axis=0)
        delta = candidate.max(axis=0) + emission[t]
    path = np.zeros(len(emission), np.int8)
    path[-1] = delta.argmax()
    for t in range(len(emission) - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return np.repeat(np.array(['H', 'A', 'D'])[path], params['stride'])[:n_frames]
