"""Symmetric local team control and binary decoding."""
import numpy as np
from . import segments as hp
from .context import mean

def local_features(pack, cfg):
    distance = pack['distance']
    relative = pack['relative']
    slots = pack['teams']
    nearest, control = ([], [])
    for team in (0, 1):
        d = distance[:, slots == team]
        v = relative[:, slots == team]
        distance_seen = np.isfinite(d).any(axis=1)
        nearest.append(np.where(distance_seen, np.where(np.isfinite(d), d, np.inf).min(axis=1), np.nan))
        valid = np.isfinite(d) & np.isfinite(v)
        likelihood = np.exp(-0.5 * ((d / cfg['contact_distance_scale']) ** 2 + (v / cfg['relative_speed_scale']) ** 2))
        control.append(np.where(valid.any(axis=1), np.where(valid, likelihood, 0.0).max(axis=1), np.nan))
    n = len(distance)
    position = np.arange(n)
    advantage = np.log1p(nearest[1]) - np.log1p(nearest[0])
    quality = control[0] - control[1]
    signed = (advantage, quality, np.tanh(advantage) * np.abs(quality))
    owner_columns = [x[np.clip(position + k, 0, n - 1)] for k in cfg['local_offsets_bins'] for x in signed]
    owner_columns += [mean(x, r) for r in cfg['local_mean_radii_bins'] for x in signed]
    owner_x = np.column_stack(owner_columns)
    return owner_x.astype(np.float32)

def decode_binary(emission, logits, initial):
    """Viterbi over H/A with constant, symmetric switch odds."""
    score = initial + emission[0]
    stay_trace = np.zeros((len(emission), 2), dtype=bool)
    for t in range(1, len(emission)):
        odds = np.clip(logits[t], -18, 8)
        switch = (score - np.logaddexp(0.0, -odds))[::-1]
        stay = score - np.logaddexp(0.0, odds)
        stays = stay >= switch
        stay_trace[t] = stays
        score = np.where(stays, stay, switch) + emission[t]
    state = int(score.argmax())
    path = np.empty(len(emission), dtype=np.int8)
    for t in range(len(emission) - 1, -1, -1):
        path[t] = state
        if not stay_trace[t, state]:
            state = 1 - state
    return path

def refine(base, hold, emission, owner_x, params, fitted):
    transition = np.asarray(params['transitions'])
    initial = np.asarray(params['initial'])
    start_index = np.searchsorted(hold, np.arange(len(emission)))
    dead = base[start_index] == 'D'
    live_mass = np.logaddexp(emission[:, 0], emission[:, 1])
    team_emission = emission[:, :2] - live_mass[:, None]
    static_switch = (transition[0, 1] / transition[0, :2].sum() + transition[1, 0] / transition[1, :2].sum()) / 2
    switch_odds = np.full((len(emission), 2), np.log(static_switch) - np.log1p(-static_switch))
    odds = np.where(np.isfinite(owner_x), owner_x, 0.0) / fitted['scales'] @ np.asarray(fitted['coefficients'])
    valid = np.isfinite(owner_x).any(axis=1)
    team_emission[valid] = np.column_stack([-np.logaddexp(0, -odds[valid]), -np.logaddexp(0, odds[valid])])
    sampled = np.full(len(emission), 'D')
    for start, stop, is_dead in hp.runs(dead):
        if not is_dead:
            path = decode_binary(team_emission[start:stop], switch_odds[start:stop], np.log(initial[:2] / initial[:2].sum()))
            sampled[start:stop] = np.array(list('HA'))[path]
    result = sampled[hold]
    if not np.array_equal(result == 'D', base == 'D'):
        raise ValueError('Team Local changed the baseline D mask')
    return result
