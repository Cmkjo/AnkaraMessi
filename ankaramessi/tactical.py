"""Read the official tactical ATD and verify its speed-unit policy."""
from dataclasses import replace
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from .tracking import Observations, sha256

def _read_export(meta_path, raw_path, team_mapping):
    root = ET.parse(meta_path).getroot()
    fps = float(root.findtext('.//FrameRate'))
    field = root.find('.//FieldSize')
    pitch = (float(field.findtext('Width')), float(field.findtext('Height')))
    if fps != 25 or not np.isfinite(pitch).all() or min(pitch) <= 0:
        raise ValueError('Invalid field/cadence')
    channels = {x.get('id'): x.findtext('Unit') for x in root.findall('.//Channel')}
    if channels != {'x': 'normalized', 'y': 'normalized', 's': 'kph'}:
        raise ValueError('Unsupported units')
    teams = {x.get('id'): team_mapping.get((x.findtext('Name') or '').strip().upper()) for x in root.findall('.//Team')}
    if set(teams.values()) != {'H', 'A'}:
        raise ValueError('Unknown team names')
    people = {x.get('id'): teams[x.get('teamId')] for x in root.findall('.//Player')}
    refs = {x.get('id'): (x.get('playerId'), x.get('channelId')) for x in root.findall('.//PlayerChannel')}
    specs = root.findall('.//DataFormatSpecification')
    if len(specs) != 1:
        raise ValueError('Only single-layout EPTS supported by this reader')
    spec = specs[0]
    start = int(spec.get('startFrame'))
    stop = int(spec.get('endFrame')) + 1
    if start != 1 or spec.get('separator') != ':':
        raise ValueError('Unsupported frame register')
    groups = spec.findall('./SplitRegister')
    if len(groups) != 2 or groups[0].get('separator') != ';' or groups[1].get('separator') != ',':
        raise ValueError('Unsupported record grouping')
    slots = []
    for group in groups[0].findall('./SplitRegister'):
        rr = [refs[x.get('playerChannelId')] for x in group.findall('PlayerChannelRef')]
        if group.get('separator') != ',' or len(rr) != 3 or [x[1] for x in rr] != ['x', 'y', 's'] or (len(set((x[0] for x in rr))) != 1):
            raise ValueError('Unsupported player channels')
        slots.append(rr[0][0])
    if len(set(slots)) != len(slots) or set(slots) != set(people):
        raise ValueError('Slot layout mismatch')
    if [x.get('channelId') for x in groups[1].findall('BallChannelRef')] != ['x', 'y', 's']:
        raise ValueError('Unsupported ball channels')
    n = stop - start
    a = np.full((n, len(slots) + 1, 3), np.nan, dtype=np.float32)
    count = 0
    with open(raw_path) as f:
        for i, line in enumerate(f):
            parts = line.strip().split(':')
            if len(parts) != 3 or i >= n or int(parts[0]) != i + 1:
                raise ValueError('Frame continuity mismatch')
            player_groups = parts[1].split(';')
            if len(player_groups) != len(slots) or any((len(g.split(',')) != 3 for g in player_groups)) or len(parts[2].split(',')) != 3:
                raise ValueError('Raw channel count mismatch')
            values = np.array([float(v) for g in player_groups + [parts[2]] for v in g.split(',')], dtype=np.float32)
            if np.isinf(values).any():
                raise ValueError('Infinite raw values')
            a[i] = values.reshape(len(slots) + 1, 3)
            count += 1
    if count != n:
        raise ValueError('Incomplete timeline')
    players = a[:, :-1, :2] * np.array(pitch, dtype=np.float32)
    ball = a[:, -1, :2] * np.array(pitch, dtype=np.float32)
    speeds = a[:, :-1, 2] / 3.6
    speeds[speeds < 0] = np.nan
    obs = Observations(np.arange(n), ball, players, speeds, np.array([0 if people[s] == 'H' else 1 for s in slots]), pitch, 25)
    return (obs, dict(frames=n, slots=len(slots), fps=25, pitch_m=list(pitch), speed_unit_xml='kph', speed_conversion='divide by 3.6 to m/s', finite_ball_frames=int(np.isfinite(ball).all(axis=1).sum()), mean_finite_players=float(np.isfinite(players).all(axis=2).sum(axis=1).mean()), atd_to_video_offset=-1, layouts=1, slot_identity_continuity_verified=False))

def load(meta_path, raw_path, team_mapping, policy):
    if policy['raw_speed_unit_override'] != 'm/s' or policy['legacy_reader_output_multiplier'] != 3.6:
        raise ValueError('Unsupported verified unit policy')
    o, audit = _read_export(meta_path, raw_path, team_mapping)
    result = replace(o, speeds=o.speeds * 3.6)
    audit.update(reader='verified_tactical_export', effective_speed_unit='m/s', speed_conversion='undo legacy division by 3.6 for exact audited export', metadata_unit_overridden=True, numeric_roundtrip_note='float32 roundtrip, original speed recovered within floating-point tolerance')
    return (result, audit)
