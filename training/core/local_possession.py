"""Local observation heads for relative control and dead-ball boundaries."""
import numpy as np
import heuristic_post as hp
import joint_model as jm
import segment_boundaries as sb
from external_context import mean

def local_features(pack, cfg):
    distance = pack['distance']
    relative = pack['relative']
    slots = pack['teams']
    nearest, control, observed_motion = ([], [], [])
    for team in (0, 1):
        d = distance[:, slots == team]
        v = relative[:, slots == team]
        distance_seen = np.isfinite(d).any(axis=1)
        nearest.append(np.where(distance_seen, np.where(np.isfinite(d), d, np.inf).min(axis=1), np.nan))
        valid = np.isfinite(d) & np.isfinite(v)
        likelihood = np.exp(-0.5 * ((d / cfg['contact_distance_scale']) ** 2 + (v / cfg['relative_speed_scale']) ** 2))
        control.append(np.where(valid.any(axis=1), np.where(valid, likelihood, 0.0).max(axis=1), np.nan))
        observed_motion.append(valid.any(axis=1))
    n = len(distance)
    position = np.arange(n)
    advantage = np.log1p(nearest[1]) - np.log1p(nearest[0])
    quality = control[0] - control[1]
    signed = (advantage, quality, np.tanh(advantage) * np.abs(quality))
    owner_columns = [x[np.clip(position + k, 0, n - 1)] for k in cfg['local_offsets_bins'] for x in signed]
    owner_columns += [mean(x, r) for r in cfg['local_mean_radii_bins'] for x in signed]
    owner_x = np.column_stack(owner_columns)
    edge = np.clip(pack['edge'], -12, 12)
    ball = np.log1p(pack['bs'])
    local = np.log1p(pack['local'])
    collective = np.log1p(pack['collective'])
    near = np.log1p(np.fmin(*nearest))
    contact = np.fmax(*control)
    base = (edge, ball, local, collective, near, contact)
    dead_columns = [np.ones(n)]
    dead_columns += [x[np.clip(position + k, 0, n - 1)] for k in cfg['local_offsets_bins'] for x in base]
    dead_columns += [mean(x, r) for r in cfg['local_mean_radii_bins'] for x in base]
    dead_columns += [np.isfinite(x) for x in base]
    dead_columns += [np.isfinite(pack['ball']).all(axis=1), observed_motion[0] | observed_motion[1]]
    return (owner_x.astype(np.float32), np.column_stack(dead_columns).astype(np.float32))

def fit(sequences, cfg):
    columns = {'team': [], 'dead': []}
    for (owner_x, dead_x), y in sequences:
        live = np.isin(y, list('HA'))
        known = live | (y == 'D')
        local = np.isfinite(owner_x).any(axis=1)
        columns['team'].append((owner_x[live & local], y[live & local] == 'H'))
        columns['dead'].append((dead_x[known], y[known] == 'D'))
    result = {}
    for name, rows in columns.items():
        x, y = (np.concatenate(part) for part in zip(*rows))
        x = np.where(np.isfinite(x), x, 0.0).astype(float)
        scale = np.maximum(np.sqrt(np.mean(x * x, axis=0)), 1e-06)
        y = y.astype(float)
        if name == 'team':
            x, y = (np.r_[x, -x], np.r_[y, 1 - y])
        head = jm.solve(x / scale, y, np.zeros(len(y)), l2=cfg['l2'])
        result[name] = dict(scales=scale.tolist(), coefficients=head['coefficients'], examples=len(y), positives=int(y.sum()))
    return result

def decode(base, hold, emission, features, switch_features, params, fitted, switch_head, method):
    """Decode one half on the fixed 5Hz grid, and preserve requested native labels."""
    if method == 'baseline':
        return base.copy()
    owner_x, dead_x = features
    transition = np.asarray(params['transitions'])
    initial = np.asarray(params['initial'])
    start_index = np.searchsorted(hold, np.arange(len(emission)))
    baseline_dead = base[start_index] == 'D'
    dead = baseline_dead
    offsets = np.zeros((2, 1))
    live_mass = np.logaddexp(emission[:, 0], emission[:, 1])
    team_emission = emission[:, :2] - live_mass[:, None]
    static_switch = (transition[0, 1] / transition[0, :2].sum() + transition[1, 0] / transition[1, :2].sum()) / 2
    switch_odds = np.full((len(emission), 2), np.log(static_switch) - np.log1p(-static_switch))
    if method in ('dead_local', 'both_local'):
        enter = transition[:2, 2].mean()
        leave = transition[2, :2].sum()
        dead_trans = np.array([enter, leave])
        logits = np.broadcast_to(np.log(dead_trans) - np.log1p(-dead_trans), (len(emission), 2))
        odds = sb.edge_logits(dead_x, fitted['dead'])
        dead_emission = np.column_stack([-np.logaddexp(0, odds), -np.logaddexp(0, -odds)])
        dead = sb.decode_binary(dead_emission, logits, np.log([initial[:2].sum(), initial[2]]), offsets) == 1
    if method == 'team_edges':
        switch_odds = sb.edge_logits(switch_features, switch_head)
    if method in ('team_local', 'both_local'):
        odds = sb.edge_logits(owner_x, fitted['team'])
        valid = np.isfinite(owner_x).any(axis=1)
        team_emission[valid] = np.column_stack([-np.logaddexp(0, -odds[valid]), -np.logaddexp(0, odds[valid])])
    sampled = np.full(len(emission), 'D')
    for start, stop, is_dead in hp.runs(dead):
        if not is_dead:
            path = sb.decode_binary(team_emission[start:stop], switch_odds[start:stop], np.log(initial[:2] / initial[:2].sum()), offsets)
            sampled[start:stop] = np.array(list('HA'))[path]
    result = sampled[hold]
    if method == 'dead_local':
        preserve = (base != 'D') & (result != 'D')
        result[preserve] = base[preserve]
    if method in ('team_edges', 'team_local') and (not np.array_equal(result == 'D', base == 'D')):
        raise ValueError('Team-only decoder changed D')
    return result
