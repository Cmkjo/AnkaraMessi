"""Velocity and distance-conditioned motion evidence."""
import numpy as np

def velocity(position, frame_ids, fps=25, span=2):
    """Least-squares slope and peak adjacent-step speed, with complete-window validity.

    A missing intermediate sample or a nonconsecutive frame invalidates the whole
    window. Peak step speed detects jumps even when the centered slope cancels.
    """
    x = np.asarray(position, dtype=float)
    ids = np.asarray(frame_ids)
    if x.ndim not in (2, 3) or x.shape[-1] != 2 or len(ids) != len(x) or (span < 1) or (fps <= 0):
        raise ValueError('Invalid motion input')
    out = np.full_like(x, np.nan)
    peak = np.full(x.shape[:-1], np.nan)
    n = len(x)
    width = 2 * span + 1
    if n < width:
        return (out, peak)
    size = n - 2 * span
    valid = np.ones(x[:size].shape[:-1], dtype=bool)
    continuous = np.ones(size, dtype=bool)
    slope = np.zeros_like(x[:size])
    maximum = np.zeros_like(peak[:size])
    offsets = np.arange(-span, span + 1)
    for j, offset in enumerate(offsets):
        block = x[j:j + size]
        valid &= np.isfinite(block).all(axis=-1)
        slope += offset * np.where(np.isfinite(block), block, 0)
        if j:
            continuous &= ids[j:j + size] - ids[j - 1:j - 1 + size] == 1
            step = np.linalg.norm(block - x[j - 1:j - 1 + size], axis=-1) * fps
            maximum = np.maximum(maximum, step)
    if x.ndim == 3:
        valid &= continuous[:, None]
    else:
        valid &= continuous
    slope *= fps / float(np.sum(offsets ** 2))
    out[span:n - span] = np.where(valid[..., None], slope, np.nan)
    peak[span:n - span] = np.where(valid, maximum, np.nan)
    return (out, peak)

def raw_motion(o, span=2):
    """Columns: relative H/A speed, ball peak step speed, selected H/A peak speed.

    No distance threshold, label, state flag or inferred possessor enters here.
    """
    bv, bp = velocity(o.ball, o.frame_ids, o.fps, span)
    pv, pp = velocity(o.players, o.frame_ids, o.fps, span)
    distances = np.linalg.norm(o.players - o.ball[:, None, :], axis=2)
    out = np.full((len(o.ball), 5), np.nan)
    out[:, 2] = bp
    row = np.arange(len(out))
    for team in (0, 1):
        indices = np.flatnonzero(o.teams == team)
        if not len(indices):
            continue
        d = np.where(np.isfinite(distances[:, indices]), distances[:, indices], np.inf)
        nearest = d.min(axis=1)
        unique = np.isfinite(nearest) & ((d == nearest[:, None]).sum(axis=1) == 1)
        selected = indices[d.argmin(axis=1)]
        rv = np.linalg.norm(pv[row, selected] - bv, axis=1)
        out[:, team] = np.where(unique, rv, np.nan)
        out[:, 3 + team] = np.where(unique, pp[row, selected], np.nan)
    return out

def motion_features(raw, limits):
    raw = np.asarray(raw, dtype=float)
    if raw.ndim != 2 or raw.shape[1] != 5:
        raise ValueError('Expected five raw motion columns')
    valid = np.isfinite(raw[:, :2]) & np.isfinite(raw[:, 2, None]) & np.isfinite(raw[:, 3:5])
    valid &= (raw[:, :2] >= 0) & (raw[:, 2, None] <= limits['ball_peak_step_cap_mps'])
    valid &= raw[:, 3:5] <= limits['player_peak_step_cap_mps']
    return np.log1p(np.where(valid, raw[:, :2], np.nan))

def control_change(p, cal, limits, cfg):
    """Bonus for a nearby player's increasing speed agreement over 0.4s."""
    d, rv = (p['distance'], p['relative'])
    k = round(p['hz'] * cfg['control_lookback_seconds'])
    valid = np.isfinite(d) & np.isfinite(rv)
    valid &= p['pp'] <= limits['player_peak_step_cap_mps']
    valid &= p['bp'][:, None] <= limits['ball_peak_step_cap_mps']
    agreement = np.exp(-0.5 * (rv / cal['sigma_v']) ** 2)
    impulse = np.zeros_like(d)
    both = valid[k:].copy()
    for j in range(k):
        both &= valid[j:len(valid) - k + j]
    impulse[k:] = np.where(both, np.exp(-0.5 * (d[k:] / cal['sigma_d']) ** 2) * np.maximum(agreement[k:] - agreement[:-k], 0), 0)
    return cfg['control_logodds_cap'] * (impulse[:, p['teams'] == 0].max(axis=1) - impulse[:, p['teams'] == 1].max(axis=1))
