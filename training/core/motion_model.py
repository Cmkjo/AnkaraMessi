"""External-only kinematic ablation; all observations and caps are label-free."""
import numpy as np
import model
import contrast_model as cm
FEATURE_NAMES = ['log1p_relative_speed_nearest_H_mps', 'log1p_relative_speed_nearest_A_mps']

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

def fit_limits(raw_sequences, quantile=0.995):
    """Fit caps from observation-only fit folds, sampled at the unchanged 5 Hz."""
    if not 0 < quantile <= 1:
        raise ValueError('Invalid quantile')
    raw = np.concatenate([np.asarray(x)[::model.STRIDE] for x in raw_sequences])
    ball = raw[:, 2]
    player = raw[:, 3:5].reshape(-1)
    ball = ball[np.isfinite(ball)]
    player = player[np.isfinite(player)]
    if min(len(ball), len(player)) < 10:
        raise ValueError('Insufficient external motion evidence')
    return dict(quantile=quantile, ball_peak_step_cap_mps=float(np.quantile(ball, quantile)), player_peak_step_cap_mps=float(np.quantile(player, quantile)), ball_samples=len(ball), player_samples=len(player))

def motion_features(raw, limits):
    raw = np.asarray(raw, dtype=float)
    if raw.ndim != 2 or raw.shape[1] != 5:
        raise ValueError('Expected five raw motion columns')
    valid = np.isfinite(raw[:, :2]) & np.isfinite(raw[:, 2, None]) & np.isfinite(raw[:, 3:5])
    valid &= (raw[:, :2] >= 0) & (raw[:, 2, None] <= limits['ball_peak_step_cap_mps'])
    valid &= raw[:, 3:5] <= limits['player_peak_step_cap_mps']
    return np.log1p(np.where(valid, raw[:, :2], np.nan))

def fit(sequences, quantile=0.995):
    """Each sequence is (original five features, targets, five raw motion columns)."""
    sequences = list(sequences)
    params = cm.fit([(f, y) for f, y, _ in sequences])
    limits = fit_limits([raw for _, _, raw in sequences], quantile)
    samples = [[] for _ in range(3)]
    for f, y, raw in sequences:
        if len(f) != len(y) or len(raw) != len(y):
            raise ValueError('Motion timeline mismatch')
        z = motion_features(raw, limits)[::model.STRIDE]
        y = np.asarray(y)[::model.STRIDE]
        for swapped in (False, True):
            zz = z[:, ::-1] if swapped else z
            yy = np.where(y == 'H', 'A', np.where(y == 'A', 'H', y)) if swapped else y
            for i, c in enumerate('HAD'):
                samples[i].append(zz[yy == c])
    for i, chunks in enumerate(samples):
        z = np.concatenate(chunks)
        count = np.isfinite(z).sum(axis=0)
        if np.any(count < 10):
            raise ValueError('Insufficient external class motion evidence')
        params['means'][i].extend(np.nanmean(z, axis=0).tolist())
        params['variances'][i].extend(np.maximum(np.nanvar(z, axis=0), 1e-06).tolist())
        params['training_feature_counts'][i].extend(count.tolist())
    params['feature_names'] = list(params['feature_names']) + FEATURE_NAMES
    params.update(schema=3, variant='external_hmm_motion_v2', motion_limits=limits)
    return params

def predict(f, raw, params):
    extra = motion_features(raw, params['motion_limits'])
    return model.predict(np.column_stack([cm.append_contrast(f), extra]), params)
