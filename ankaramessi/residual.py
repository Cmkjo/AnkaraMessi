"""Joint distance and relative-motion correction for H/A."""
import numpy as np
from . import geometry as cm, kinematics as mm, hmm as nm

def features(f, raw, params):
    f = np.asarray(f)
    motion = mm.motion_features(raw, params['motion_limits'])
    valid = np.isfinite(f[:, :2]).all(axis=1) & (f[:, :2] >= 0).all(axis=1) & np.isfinite(motion).all(axis=1)
    distance = np.log1p(np.where(f[:, :2] >= 0, f[:, :2], np.nan))
    x = np.column_stack([distance[:, 1] - distance[:, 0], motion[:, 1] - motion[:, 0], distance[:, 1] * motion[:, 1] - distance[:, 0] * motion[:, 0]])
    x[~valid] = np.nan
    return x

def base_emissions(f, params):
    mu = np.asarray(params['means'])
    var = np.asarray(params['variances'])
    return nm.gaussian_emissions(cm.append_contrast(f)[::params['stride']], mu[:, :6], var[:, :6])

def frozen_emissions(f, raw, params, joint):
    base = base_emissions(f, params)
    x = features(f, raw, params)[::params['stride']]
    valid = np.isfinite(x).all(axis=1)
    delta = np.zeros(len(x))
    delta[valid] = x[valid] / joint['scales'] @ np.asarray(joint['coefficients'])
    return nm.fuse(base, delta)
