"""Signed nearest-team distance feature and external parameter estimation."""
import numpy as np
import model
FEATURE_NAME = 'distance_advantage_H_m'

def append_contrast(f):
    f = np.asarray(f, dtype=float)
    if f.ndim != 2 or f.shape[1] != 5:
        raise ValueError('Expected original five features')
    good = np.isfinite(f[:, :2]).all(axis=1)
    advantage = np.full(len(f), np.nan)
    advantage[good] = f[good, 1] - f[good, 0]
    return np.column_stack([f, advantage])

def fit(sequences):
    sequences = list(sequences)
    params = model.fit(sequences)
    samples = [[] for _ in range(3)]
    for f, y in sequences:
        z = append_contrast(f)[::model.STRIDE, -1]
        y = np.asarray(y)[::model.STRIDE]
        for swapped in (False, True):
            zz = -z if swapped else z
            yy = np.where(y == 'H', 'A', np.where(y == 'A', 'H', y)) if swapped else y
            for i, c in enumerate('HAD'):
                samples[i].append(zz[yy == c])
    for i, chunks in enumerate(samples):
        z = np.concatenate(chunks)
        n = int(np.isfinite(z).sum())
        if n < 10:
            raise ValueError('Insufficient external contrast evidence')
        params['means'][i].append(float(np.nanmean(z)))
        params['variances'][i].append(float(max(np.nanvar(z), 1e-06)))
        params['training_feature_counts'][i].append(n)
    params['feature_names'] = list(model.FEATURES) + [FEATURE_NAME]
    params.update(schema=2, variant='external_hmm_contrast_v1')
    return params

def predict(f, params):
    return model.predict(append_contrast(f), params)
