"""Player activity, proximity, occlusions and team shape."""
import warnings
import numpy as np
from . import kinematics as mm

def mean(x, radius):
    x = np.asarray(x)
    n = len(x)
    t = np.arange(n)
    a = np.maximum(0, t - radius)
    b = np.minimum(n, t + radius + 1)
    finite = np.isfinite(x)
    count = np.r_[0, np.cumsum(finite)]
    total = np.r_[0, np.cumsum(np.where(finite, x, 0.0))]
    den = count[b] - count[a]
    return np.divide(total[b] - total[a], den, out=np.full(n, np.nan), where=den > 0)

def pack(o, indices):
    """All-player control fields kept at 5Hz; native velocity windows first."""
    bv, bpeak = mm.velocity(o.ball, o.frame_ids, o.fps, 2)
    pv, ppeak = mm.velocity(o.players, o.frame_ids, o.fps, 2)
    ball = o.ball[indices]
    pos = o.players[indices]
    speed = o.speeds[indices]
    v = pv[indices]
    bvel = bv[indices]
    distance = np.linalg.norm(pos - ball[:, None, :], axis=-1)
    relative = np.linalg.norm(v - bvel[:, None, :], axis=-1)
    good = np.isfinite(speed) & (speed >= 0)
    near = good & (distance <= 12)

    def avg(mask):
        n = mask.sum(axis=1)
        return np.divide(np.where(mask, speed, 0).sum(axis=1), n, out=np.full(len(ball), np.nan), where=n >= 2)
    local = avg(near)
    collective = avg(good)
    bs = np.linalg.norm(bvel, axis=1)
    edge = np.min(np.concatenate([ball, np.asarray(o.pitch) - ball], axis=1), axis=1)
    ctx = []
    centers = []
    deep = np.full((len(ball), 2), -1, int)
    for team in [0, 1]:
        slots = np.flatnonzero(o.teams == team)
        pp = pos[:, slots]
        ss = speed[:, slots]
        count = np.isfinite(pp).all(axis=2).sum(axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            center = np.nanmean(pp, axis=1)
            disp = np.nanstd(pp, axis=1)
            ms = np.nanmean(np.where(ss >= 0, ss, np.nan), axis=1)
        xx = np.column_stack([np.abs(center[:, 0] - o.pitch[0] / 2) / o.pitch[0], np.abs(center[:, 1] - o.pitch[1] / 2) / o.pitch[1], disp[:, 0] / o.pitch[0], disp[:, 1] / o.pitch[1], np.log1p(ms)])
        xx[count < 5] = np.nan
        ctx.append(xx)
        centers.append(center)
    home_low = np.nanmedian(centers[0][:, 0]) < np.nanmedian(centers[1][:, 0])
    for team in [0, 1]:
        slots = np.flatnonzero(o.teams == team)
        pp = pos[:, slots]
        low = home_low if team == 0 else not home_low
        x = np.where(np.isfinite(pp[:, :, 0]), pp[:, :, 0], np.inf if low else -np.inf)
        deep[:, team] = slots[x.argmin(axis=1) if low else x.argmax(axis=1)]
    return dict(distance=distance.astype(np.float32), relative=relative.astype(np.float32), bp=bpeak[indices], pp=ppeak[indices], local=local, collective=collective, bs=bs, edge=edge, ball=ball, teams=o.teams, pitch=o.pitch, deep=deep, context=ctx[1] - ctx[0], defends_low=[bool(home_low), not bool(home_low)], hz=5)

def features(p, cal, params):
    d = p['distance']
    rv = p['relative']
    caps = params['motion_limits']
    valid = np.isfinite(d) & np.isfinite(rv) & np.isfinite(p['pp']) & np.isfinite(p['bp'][:, None])
    valid &= (p['pp'] <= caps['player_peak_step_cap_mps']) & (p['bp'][:, None] <= caps['ball_peak_step_cap_mps'])
    q = np.where(valid, np.exp(-0.5 * (d / cal['sigma_d']) ** 2 - 0.5 * (rv / cal['sigma_v']) ** 2), 0.0)
    quality = np.column_stack([q[:, p['teams'] == t].max(axis=1) for t in [0, 1]])
    qdiff = (quality[:, 0] - quality[:, 1])[:, None]
    seen = np.isfinite(p['ball']).all(axis=1)
    local = np.log1p(p['local'])
    bs = np.log1p(p['bs'])
    col = np.log1p(p['collective'])
    activity = np.column_stack([np.where(seen, mean(x, r), np.nan) for r in [2, 10] for x in [local, bs, col]])
    keeper = np.zeros(len(seen))
    row = np.arange(len(seen))
    for t in [0, 1]:
        depth = p['ball'][:, 0] if p['defends_low'][t] else p['pitch'][0] - p['ball'][:, 0]
        in_box = (depth <= 16.5) & (np.abs(p['ball'][:, 1] - p['pitch'][1] / 2) <= 20.16) & (p['edge'] >= 0)
        j = p['deep'][:, t]
        keeper = np.maximum(keeper, (in_box & (d[row, j] <= cal['sigma_d']) & (rv[row, j] <= cal['sigma_v']) & valid[row, j]).astype(float))
    gap = np.zeros((len(seen), 3))
    edges = np.r_[0, np.flatnonzero(seen[1:] != seen[:-1]) + 1, len(seen)]
    for a, b in zip(edges[:-1], edges[1:]):
        if seen[a]:
            continue
        exit_ok = a > 0 and np.isfinite(p['edge'][a - 1]) and (p['edge'][a - 1] <= cal['sigma_d'])
        back_ok = b < len(seen) and np.isfinite(p['edge'][b]) and (p['edge'][b] <= cal['sigma_d'])
        if not (exit_ok or back_ok):
            continue
        decay = np.exp(-np.arange(b - a) / 5 / cal['gap_tau'])
        gap[a:b, 0] = decay * exit_ok
        gap[a:b, 1] = decay * back_ok
        gap[a:b, 2] = np.where(np.isfinite(col[a:b]), decay / (1 + np.expm1(col[a:b])), 0.0)
    return dict(activity=activity, keeper=keeper[:, None], gap=gap, context=p['context'], control=qdiff, multi=np.column_stack([activity, keeper, gap, quality.max(axis=1)]), quality=quality)

def predict_logit(x, head):
    return np.where(np.isfinite(x), x, 0.0) / head['scales'] @ np.asarray(head['coefficients'])
