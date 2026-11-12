"""Player activity, team context and control evidence.

Parameter estimation uses labelled external sequences through fit().
Feature extraction and prediction use observations only."""
import warnings
import numpy as np
import motion_model as mm, joint_model as jm, normalized_model as nm

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

def scales(seqs):
    distances = []
    relatives = []
    dead_lengths = []
    for s in seqs:
        f = s['f'][::5]
        y = s['y'][::5]
        raw = s['raw'][::5]
        for team, c in enumerate('HA'):
            use = y == c
            distances.extend(f[use, team].tolist())
            relatives.extend(raw[use, team].tolist())
        edges = np.r_[0, np.flatnonzero(y[1:] != y[:-1]) + 1, len(y)]
        dead_lengths.extend(((b - a) / 5 for a, b in zip(edges[:-1], edges[1:]) if y[a] == 'D'))

    def quantile(x):
        a = np.asarray(x)
        a = a[np.isfinite(a) & (a >= 0)]
        if len(a) < 20:
            raise ValueError('Insufficient scale data')
        return max(0.25, float(np.quantile(a, 0.25)))
    return dict(sigma_d=quantile(distances), sigma_v=quantile(relatives), gap_tau=max(0.25, float(np.median(dead_lengths))))

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

def fit_logistic(xs, ys, offsets, symmetric=False):
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    offset = np.concatenate(offsets)
    x = np.where(np.isfinite(x), x, 0.0)
    rms = np.maximum(np.sqrt(np.mean(x * x, axis=0)), 1e-06)
    x = x / rms
    if symmetric:
        x = np.r_[x, -x]
        y = np.r_[y, 1 - y]
        offset = np.r_[offset, -offset]
    fitted = jm.solve(x, y, offset, l2=0.001)
    fitted['scales'] = rms.tolist()
    return fitted

def predict_logit(x, head):
    return np.where(np.isfinite(x), x, 0.0) / head['scales'] @ np.asarray(head['coefficients'])

def frozen_emissions(f, raw, params, joint):
    base = jm.base_emissions(f, params)
    x = jm.features(f, raw, params)[::params['stride']]
    valid = np.isfinite(x).all(axis=1)
    delta = np.zeros(len(x))
    delta[valid] = x[valid] / joint['scales'] @ np.asarray(joint['coefficients'])
    return nm.fuse(base, delta)

def fit(seqs, params, joint, pseudo):
    cal = scales(seqs)
    dead = {key: ([], [], []) for key in ['activity', 'keeper', 'gap', 'multi']}
    control = ([], [], [])
    context = ([], [], [])
    for s in seqs:
        for entry in s['entries'][:2]:
            f, raw, p = (entry['f'], entry['raw'], entry['pack'])
            y = s['y'][::5]
            e = frozen_emissions(f, raw, params, joint)
            ff = features(p, cal, params)
            offset = e[:, 2] - np.logaddexp(e[:, 0], e[:, 1])
            known = np.isin(y, list('HAD'))
            for key, chunks in dead.items():
                chunks[0].append(ff[key][known])
                chunks[1].append((y[known] == 'D').astype(float))
                chunks[2].append(offset[known])
            if entry['scenario'] == 'clean':
                use = np.isin(y, list('HA'))
                control[0].append(ff['control'][use])
                control[1].append((y[use] == 'H').astype(float))
                control[2].append((e[:, 0] - e[:, 1])[use])
                labels = pseudo[s['id'], s['half']]
                valid = np.isin(labels, list('HA')) & np.isfinite(ff['context']).all(axis=1)
                context[0].append(ff['context'][valid])
                context[1].append((labels[valid] == 'H').astype(float))
                context[2].append(np.zeros(valid.sum()))
    return dict(calibration=cal, dead={k: fit_logistic(*v) for k, v in dead.items()}, control=fit_logistic(*control, symmetric=True), context=fit_logistic(*context, symmetric=True))

def predict(f, raw, p, params, joint, fitted, method):
    e = frozen_emissions(f, raw, params, joint)
    ff = features(p, fitted['calibration'], params)
    delta = np.zeros(len(e))
    dd = np.zeros(len(e))
    if method in ['context', 'multi']:
        x = ff['context']
        complete = np.isfinite(x).all(axis=1)
        context = predict_logit(x, fitted['context'])
        context[~complete] = 0
        delta += np.clip(context, -3, 3) * (1 - ff['quality'].max(axis=1))
    if method in ['control', 'multi']:
        delta += np.clip(predict_logit(ff['control'], fitted['control']), -3, 3)
    if method in ['activity', 'keeper', 'gap', 'multi']:
        dd = np.clip(predict_logit(ff[method], fitted['dead'][method]), -8, 8)
        if method == 'keeper':
            dd = np.minimum(dd, 0)
    result = nm.fuse(e, np.clip(delta, -3, 3))
    result[:, 2] += dd
    return (nm.decode(result, params, len(f)), dict(max_abs_ha_delta=float(np.max(np.abs(np.clip(delta, -3, 3)))), max_abs_dead_delta=float(np.max(np.abs(dd))), nonzero_dead=int(np.sum(dd != 0)), nonzero_ha=int(np.sum(delta != 0))))
