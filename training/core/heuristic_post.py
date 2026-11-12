"""Fixed, observation-only postprocessor; no labels, fitting or minimum possession rule."""
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

def memory(pred, f, fps, c, ball_missing):
    out = np.array(pred, copy=True)
    controller = None
    last = -np.inf
    for i, current in enumerate(pred):
        if current == 'D':
            controller = None
            last = -np.inf
            continue
        d = f[i, :2]
        if np.isfinite(d).all():
            team = 0 if current == 'H' else 1 if current == 'A' else None
            if team is not None and d[team] <= c['contact_distance_m'] and (d[1 - team] - d[team] >= c['contact_advantage_m']):
                controller = current
                last = i
        elif ball_missing[i] and current in ('H', 'A') and (controller is not None):
            age = (i - last) / fps
            weight = np.exp(-age / c['tau_seconds'])
            if age <= c['max_seconds'] and weight >= c['min_weight']:
                out[i] = controller
    return out

def motion_context(o, f, c):
    speeds = np.asarray(o.speeds)
    good = np.isfinite(speeds) & (speeds >= 0)
    n = good.sum(axis=1)
    collective = np.divide(np.where(good, speeds, 0).sum(axis=1), n, out=np.full(len(f), np.nan), where=n >= c['min_collective_players'])
    distance = np.linalg.norm(o.players - o.ball[:, None, :], axis=-1)
    local = good & np.isfinite(distance) & (distance <= c['local_radius_m'])
    nlocal = local.sum(axis=1)
    local_speed = np.divide(np.where(local, speeds, 0).sum(axis=1), nlocal, out=np.full(len(f), np.nan), where=nlocal >= c['min_local_players'])
    return dict(ball_speed=np.expm1(f[:, 3]), local_speed=local_speed, collective_speed=collective)

def dead_guard(pred, f, fps, c, context):
    out = np.array(pred, copy=True)
    last_dead = -np.inf
    for i, current in enumerate(pred):
        if current == 'D':
            last_dead = i
            continue
        if (i - last_dead) / fps > c['max_seconds']:
            continue
        ball = context['ball_speed'][i]
        local = context['local_speed'][i]
        collective = context['collective_speed'][i]
        if np.isfinite(ball) and ball >= c['clear_ball_mps'] or (np.isfinite(local) and local >= c['clear_players_mps']) or (np.isfinite(collective) and collective >= c['clear_players_mps']):
            last_dead = -np.inf
            continue
        d = f[i, :2]
        if current in ('H', 'A') and np.isfinite(d).all() and (abs(d[0] - d[1]) <= c['ambiguity_distance_m']) and np.isfinite([ball, local, collective]).all() and (ball <= c['stationary_ball_mps']) and (local <= c['stationary_local_mps']) and (collective <= c['stationary_collective_mps']):
            out[i] = 'D'
    return out

def apply(pred, f, o, config, method):
    pred = np.asarray(pred)
    if len(pred) != len(f) or len(pred) != len(o.ball) or (not np.isin(pred, list('HADX')).all()):
        raise ValueError('Invalid postprocessor input')
    context = motion_context(o, f, config['dead_guard'])
    if method == 'merge':
        return merge(pred, f, o.fps, config['merge'])
    if method == 'memory':
        return memory(pred, f, o.fps, config['memory'], ~np.isfinite(o.ball).all(axis=1))
    if method == 'dead_guard':
        return dead_guard(pred, f, o.fps, config['dead_guard'], context)
    if method == 'combined':
        out = merge(pred, f, o.fps, config['merge'])
        out = memory(out, f, o.fps, config['memory'], ~np.isfinite(o.ball).all(axis=1))
        return dead_guard(out, f, o.fps, config['dead_guard'], context)
    raise ValueError('Unknown rule')

def restart_hold(pred, f, fps, windows, hold_seconds=3.0):
    """windows=(start,stop,H/A) indices for a single half, NOT target events.

    Anchor requires a visible ball starting to move and close tagged-team player.
    No D fabrication. Base D or extreme opposing control revokes the hold.
    """
    votes = np.zeros(len(pred), dtype=np.int8)
    speed = np.expm1(f[:, 3])
    for start, stop, team in windows:
        if team not in ('H', 'A'):
            continue
        j = 0 if team == 'H' else 1
        for i in range(max(1, start), min(stop, len(pred))):
            if np.isfinite(speed[i - 1:i + 1]).all() and speed[i - 1] < 1 and (speed[i] >= 2) and np.isfinite(f[i, :2]).all() and (f[i, j] <= 2) and (f[i, 1 - j] - f[i, j] >= 1):
                for k in range(i, min(stop, len(pred), i + int(np.ceil(hold_seconds * fps)))):
                    if pred[k] == 'D' or (np.isfinite(f[k, :2]).all() and f[k, 1 - j] <= 1.5 and (f[k, j] - f[k, 1 - j] >= 1)):
                        break
                    votes[k] |= 1 if team == 'H' else 2
                break
    out = np.array(pred, copy=True)
    out[votes == 1] = 'H'
    out[votes == 2] = 'A'
    return out
