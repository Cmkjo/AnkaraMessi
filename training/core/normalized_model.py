"""Motion evidence changes team odds without changing the frame's alive mass."""
import numpy as np
import contrast_model as cm
import motion_model as mm

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

def emissions(f, raw, params):
    stride = params['stride']
    x = cm.append_contrast(f)[::stride]
    extra = mm.motion_features(raw, params['motion_limits'])[::stride]
    mu = np.asarray(params['means'])
    var = np.asarray(params['variances'])
    base = gaussian_emissions(x, mu[:, :6], var[:, :6])
    motion = gaussian_emissions(extra, mu[:, -2:], var[:, -2:])
    delta = motion[:, 0] - motion[:, 1]
    fused = fuse(base, delta)
    audit = dict(n_sampled=len(base), n_motion_odds_nonzero=int(np.sum(delta != 0)), max_abs_alive_logmass_error=float(np.max(np.abs(np.logaddexp(fused[:, 0], fused[:, 1]) - np.logaddexp(base[:, 0], base[:, 1])))), dead_emission_exactly_preserved=bool(np.array_equal(fused[:, 2], base[:, 2])))
    return (fused, audit)

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

def predict(f, raw, params):
    e, audit = emissions(f, raw, params)
    return (decode(e, params, len(f)), audit)
