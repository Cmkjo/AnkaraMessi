"""Predict one possession label for every frame of the tactical video."""
import argparse
import csv
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
import numpy as np
from . import context, geometry, hmm, kinematics, ownership, residual, segments, tactical
from .tracking import Observations, sha256
ROOT = Path(__file__).resolve().parents[1]

def predict_half(obs, settings):
    params = settings['params']
    fitted = settings['external_fit']
    calibration = fitted['calibration']
    frame_features = geometry.features(obs)
    motion = kinematics.raw_motion(obs, 2)
    pack = context.pack(obs, np.arange(0, len(frame_features), params['stride']))
    emission = residual.frozen_emissions(frame_features, motion, params, settings['joint'])
    signals = context.features(pack, calibration, params)
    shape = context.predict_logit(signals['context'], fitted['context'])
    shape[~np.isfinite(signals['context']).all(axis=1)] = 0
    odds = np.clip(shape, -3, 3) * (1 - signals['quality'].max(axis=1))
    odds += np.clip(context.predict_logit(signals['control'], fitted['control']), -3, 3)
    change = kinematics.control_change(pack, calibration, params['motion_limits'], settings['control'])
    emission = hmm.fuse(emission, np.clip(odds + change, -3, 3))
    emission[:, 2] += np.clip(context.predict_logit(signals['multi'], fitted['dead']['multi']), -8, 8)
    labels = hmm.decode(emission, params, len(frame_features))
    baseline = segments.merge(labels, frame_features, obs.fps, settings['merge'])
    hold = np.arange(len(frame_features)) // params['stride']
    local = ownership.local_features(pack, settings['ownership_features'])
    return ownership.refine(baseline, hold, emission, local, params, settings['ownership_head'])

def predict_match(obs, settings):
    if len(obs.ball) != settings['n_frames']:
        raise ValueError('Unexpected number of video frames')
    labels = np.full(len(obs.ball), 'X')
    inside = np.zeros(len(labels), dtype=bool)
    for _, start, stop in settings['halves']:
        span = slice(start, stop)
        half = Observations(obs.frame_ids[span], obs.ball[span], obs.players[span], obs.speeds[span], obs.teams, obs.pitch, obs.fps)
        labels[span] = predict_half(half, settings)
        inside[span] = True
    if not np.isin(labels[inside], list('HAD')).all() or not np.all(labels[~inside] == 'X'):
        raise ValueError('Unexpected label or half coverage')
    return labels

def write_csv(path, labels):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode='w', encoding='utf-8', newline='', dir=path.parent, delete=False) as file:
        temporary = Path(file.name)
        try:
            writer = csv.writer(file, lineterminator='\n')
            writer.writerow(('frame', 'label'))
            writer.writerows(enumerate(labels))
            file.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tactical-dir', type=Path, default=ROOT / 'data/tactical')
    parser.add_argument('--output', type=Path, default=ROOT / 'evaluation/results/AnkaraMessi.csv')
    args = parser.parse_args()
    settings = json.loads(Path(__file__).with_name('model_parameters.json').read_text())
    directory = args.tactical_dir
    obs, _ = tactical.load(directory / 'BAR-ATH_tactical_atd.xml', directory / 'BAR-ATH_tactical_atd.txt', settings['teams'], settings['speed_policy'])
    write_csv(args.output, predict_match(obs, settings))
    print(f'{sha256(args.output)}  {args.output}')
if __name__ == '__main__':
    main()
