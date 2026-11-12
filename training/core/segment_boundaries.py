"""Factorized possession decoder with externally learned boundary hazards."""
import numpy as np
import heuristic_post as hp
import joint_model as jm
import normalized_model as nm

def observations(pack, radius):
    """Cinematic evidence at t and on either side of the candidate boundary."""
    d = pack['distance']
    rv = pack['relative']
    team_distances, team_relative = ([], [])
    for team in (0, 1):
        columns = pack['teams'] == team
        distances = np.where(np.isfinite(d[:, columns]), d[:, columns], np.inf)
        nearest = distances.argmin(axis=1)
        rows = np.arange(len(d))
        value = distances[rows, nearest]
        team_distances.append(np.where(np.isfinite(value), value, np.nan))
        team_relative.append(rv[:, columns][rows, nearest])
    distance = np.column_stack(team_distances)
    relative = np.column_stack(team_relative)
    pos = np.arange(len(d))
    offsets = (-radius, 0, radius)
    fields = np.column_stack([np.clip(pack['edge'], -8, 8), np.log1p(pack['bs']), np.log1p(pack['local']), np.log1p(pack['collective']), np.log1p(np.minimum(distance[:, 0], distance[:, 1])), np.log1p(np.minimum(relative[:, 0], relative[:, 1])), np.isfinite(pack['ball']).all(axis=1)])

    def stack(x):
        parts = [x[np.clip(pos + offset, 0, len(pos) - 1)] for offset in offsets]
        return np.column_stack([np.ones(len(pos)), *parts])
    team_features = []
    for source in (0, 1):
        target = 1 - source
        x = np.column_stack([fields, np.log1p(distance[:, source]), np.log1p(distance[:, target]), np.log1p(relative[:, source]), np.log1p(relative[:, target])])
        team_features.append(stack(x))
    return (stack(fields), np.stack(team_features, axis=1))

def age(y):
    t = np.arange(len(y))
    start = np.r_[True, y[1:] != y[:-1]]
    return t - np.maximum.accumulate(np.where(start, t, 0)) + 1

def fit(sequences, cfg):
    """Fit conditional transitions from known adjacent samples in the fit matches."""
    starts = np.asarray(cfg['age_starts_bins'])
    chunks = {k: [] for k in ('enter_D', 'exit_D', 'switch')}
    for (dead_x, team_x), labels in sequences:
        y = np.asarray(labels)
        valid = np.isin(y, list('HAD'))
        pair = valid[:-1] & valid[1:]
        before, after = (y[:-1], y[1:])
        live_age = age(np.where(y == 'D', 'D', np.where(valid, 'L', 'X')))[:-1]
        team_age = age(y)[:-1]
        for key, mask, x, target, ages in (('enter_D', pair & (before != 'D'), dead_x[1:], after == 'D', live_age), ('exit_D', pair & (before == 'D'), dead_x[1:], after != 'D', live_age), ('switch', pair & (before != 'D') & (after != 'D'), team_x[np.arange(1, len(y)), (before == 'A').astype(int)], after != before, team_age)):
            chunks[key].append((x[mask], target[mask], ages[mask]))
    result = {}
    for key, values in chunks.items():
        x, y, ages = (np.concatenate(parts) for parts in zip(*values))
        y = y.astype(float)
        x = np.where(np.isfinite(x), x, 0.0)
        scale = np.maximum(np.sqrt(np.mean(x * x, axis=0)), 1e-06)
        x = x / scale
        fit = jm.solve(x, y, np.zeros(len(y)), l2=cfg['l2'])
        odds = x @ np.asarray(fit['coefficients'])
        bucket = np.searchsorted(starts, ages, side='right') - 1
        offset = np.zeros(len(starts))
        for i in range(len(starts)):
            use = bucket == i
            z, target = (odds[use], y[use])
            prior = cfg['age_prior_exposures']
            for _ in range(30):
                p = np.exp(-np.logaddexp(0.0, -z - offset[i]))
                step = ((p - target).sum() + prior * offset[i]) / (np.sum(p * (1 - p)) + prior)
                offset[i] -= step
                if abs(step) < 1e-08:
                    break
        result[key] = dict(coefficients=fit['coefficients'], scales=scale.tolist(), age_offset=offset.tolist(), exposure=len(y), positives=int(y.sum()))
    return result

def edge_logits(x, head):
    return np.where(np.isfinite(x), x, 0.0) / head['scales'] @ np.asarray(head['coefficients'])

def decode_binary(emission, logits, initial, offsets):
    """Exact Viterbi with a final age bin that accumulates longer segments."""
    ages = offsets.shape[1]
    score = np.full((2, ages), -np.inf)
    score[:, 0] = initial + emission[0]
    switch_age = np.zeros((len(emission), 2), dtype=np.int16)
    tail_stay = np.zeros((len(emission), 2), dtype=bool)
    for t in range(1, len(emission)):
        odds = np.clip(logits[t, :, None] + offsets, -18, 8)
        switch = score - np.logaddexp(0.0, -odds)
        stay = score - np.logaddexp(0.0, odds)
        best_age = switch.argmax(axis=1)
        switch_age[t] = best_age
        incoming = switch[np.arange(2), best_age][::-1]
        if ages == 1:
            use_stay = stay[:, 0] >= incoming
            tail_stay[t] = use_stay
            score[:, 0] = np.where(use_stay, stay[:, 0], incoming)
        else:
            tail_stay[t] = stay[:, -1] >= stay[:, -2]
            score[:, 1:] = stay[:, :-1]
            score[:, -1] = np.maximum(stay[:, -1], stay[:, -2])
            score[:, 0] = incoming
        score += emission[t, :, None]
    state, elapsed = np.unravel_index(score.argmax(), score.shape)
    out = np.empty(len(emission), dtype=np.int8)
    for t in range(len(emission) - 1, -1, -1):
        out[t] = state
        if elapsed == 0:
            if ages != 1 or not tail_stay[t, state]:
                state = 1 - state
                elapsed = switch_age[t, state]
        elif elapsed < ages - 1 or not tail_stay[t, state]:
            elapsed -= 1
    return out

def decode(emission, x, params, fitted, cfg, method):
    transition = np.asarray(params['transitions'])
    initial = np.asarray(params['initial'])
    static = np.array([(transition[0, 2] + transition[1, 2]) / 2, transition[2, :2].sum(), (transition[0, 1] / transition[0, :2].sum() + transition[1, 0] / transition[1, :2].sum()) / 2])
    static_odds = np.log(static) - np.log1p(-static)
    dead_x, team_x = x
    if method == 'factorized':
        dead_logits = np.broadcast_to(static_odds[:2], (len(emission), 2))
        team_logits = np.full((len(emission), 2), static_odds[2])
    else:
        dead_logits = np.column_stack([edge_logits(dead_x, fitted[k]) for k in ('enter_D', 'exit_D')])
        team_logits = edge_logits(team_x, fitted['switch'])
    if method == 'learned_duration':
        starts = np.asarray(cfg['age_starts_bins'])
        bucket = np.searchsorted(starts, np.arange(1, starts[-1] + 1), side='right') - 1
        dead_age = np.stack([np.asarray(fitted[k]['age_offset'])[bucket] for k in ('enter_D', 'exit_D')])
        team_age = np.tile(np.asarray(fitted['switch']['age_offset'])[bucket], (2, 1))
    else:
        dead_age = team_age = np.zeros((2, 1))
    alive = np.logaddexp(emission[:, 0], emission[:, 1])
    dead = decode_binary(np.column_stack([alive, emission[:, 2]]), dead_logits, np.log([initial[:2].sum(), initial[2]]), dead_age)
    result = np.full(len(emission), 'D')
    for a, b, state in hp.runs(dead):
        if state == 0:
            owner = decode_binary(emission[a:b, :2] - alive[a:b, None], team_logits[a:b], np.log(initial[:2] / initial[:2].sum()), team_age)
            result[a:b] = np.array(list('HA'))[owner]
    return result

def step17_emissions(prepared):
    base, delta, dead, _, change = prepared
    e = nm.fuse(base, np.clip(delta + change, -3, 3))
    e[:, 2] += dead
    return e
