"""Joint distance/motion evidence as a logistic correction to fixed H/A odds."""
import numpy as np
import contrast_model as cm
import motion_model as mm
import normalized_model as nm

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

def solve(x, y, offset, l2=0.001, max_iterations=100, tolerance=1e-08):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    offset = np.asarray(offset, dtype=float)
    if x.ndim != 2 or len(x) != len(y) or offset.shape != y.shape or (l2 <= 0) or (not len(y)):
        raise ValueError('Invalid logistic inputs')
    if not np.isfinite(x).all() or not np.isfinite(offset).all() or (not np.isin(y, [0, 1]).all()):
        raise ValueError('Nonfinite logistic data')
    beta = np.zeros(x.shape[1])
    n = len(y)

    def objective(b):
        z = offset + x @ b
        return float(np.mean(np.logaddexp(0, np.where(y == 1, -z, z))) + 0.5 * l2 * np.dot(b, b))
    initial = objective(beta)
    converged = False
    for iteration in range(max_iterations):
        z = offset + x @ beta
        p = np.exp(-np.logaddexp(0, -z))
        gradient = x.T @ (p - y) / n + l2 * beta
        if np.max(np.abs(gradient)) <= tolerance:
            converged = True
            break
        hessian = (x * (p * (1 - p))[:, None]).T @ x / n + l2 * np.eye(x.shape[1])
        direction = np.linalg.solve(hessian, gradient)
        step = 1.0
        old = objective(beta)
        for _ in range(30):
            proposal = beta - step * direction
            if objective(proposal) <= old - 0.0001 * step * np.dot(gradient, direction):
                break
            step *= 0.5
        else:
            raise ValueError('Logistic line search failed')
        beta = proposal
    z = offset + x @ beta
    p = np.exp(-np.logaddexp(0, -z))
    gradient = x.T @ (p - y) / n + l2 * beta
    converged = converged or bool(np.max(np.abs(gradient)) <= tolerance)
    if not converged:
        raise ValueError('Logistic solver did not converge')
    return dict(coefficients=beta.tolist(), l2=l2, n_augmented=n, iterations=iteration + 1, initial_objective=initial, final_objective=objective(beta), max_abs_gradient=float(np.max(np.abs(gradient))), converged=converged)

def fit(sequences, params, l2=0.001, max_iterations=100, tolerance=1e-08):
    xs = []
    ys = []
    offsets = []
    for f, y, raw in sequences:
        x = features(f, raw, params)[::params['stride']]
        y = np.asarray(y)[::params['stride']]
        e = base_emissions(f, params)
        use = np.isin(y, ['H', 'A']) & np.isfinite(x).all(axis=1)
        xs.append(x[use])
        ys.append((y[use] == 'H').astype(float))
        offsets.append((e[:, 0] - e[:, 1])[use])
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    offset = np.concatenate(offsets)
    if len(y) < 20:
        raise ValueError('Insufficient alive training evidence')
    scale = np.sqrt(np.mean(x * x, axis=0))
    scale = np.maximum(scale, 1e-06)
    x = x / scale
    result = solve(np.concatenate([x, -x]), np.r_[y, 1 - y], np.r_[offset, -offset], l2, max_iterations, tolerance)
    result.update(scales=scale.tolist(), n_observed_alive=len(y), fit_target='Alive H/A only; excludes D and X; no duration filtering', intercept=0.0)
    return result

def predict(f, raw, params, joint):
    base = base_emissions(f, params)
    x = features(f, raw, params)[::params['stride']]
    valid = np.isfinite(x).all(axis=1)
    delta = np.zeros(len(x))
    delta[valid] = x[valid] / np.asarray(joint['scales']) @ np.asarray(joint['coefficients'])
    fused = nm.fuse(base, delta)
    audit = dict(n_sampled=len(x), n_complete_joint_features=int(valid.sum()), n_motion_odds_nonzero=int(np.sum(delta != 0)), max_abs_alive_logmass_error=float(np.max(np.abs(np.logaddexp(fused[:, 0], fused[:, 1]) - np.logaddexp(base[:, 0], base[:, 1])))), dead_emission_exactly_preserved=bool(np.array_equal(fused[:, 2], base[:, 2])))
    return (nm.decode(fused, params, len(f)), audit)
