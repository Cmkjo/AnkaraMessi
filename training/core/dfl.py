"""DFL external data adapter. Never imports challenge data or labels.

DFL coordinates are metres about pitch centre; speed S is km/h.
Ball status/possession are targets only, not part of Observations.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import xml.etree.ElementTree as ET
import numpy as np

@dataclass
class Observations:
    frame_ids: np.ndarray
    ball: np.ndarray
    players: np.ndarray
    speeds: np.ndarray
    teams: np.ndarray
    pitch: tuple
    fps: int = 25

@dataclass
class Half:
    name: str
    observed: Observations
    targets: np.ndarray

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def locate(root, match_id, kind):
    paths = list(Path(root).glob(f'*{kind}*DFL-MAT-{match_id}.xml'))
    if len(paths) != 1:
        raise ValueError('Expected exactly one DFL input of requested kind')
    return paths[0]

def labels(status, owner):
    result = np.full(len(status), 'X', dtype='<U1')
    result[status == 0] = 'D'
    result[(status == 1) & (owner == 1)] = 'H'
    result[(status == 1) & (owner == 2)] = 'A'
    return result

def load(root, match_id):
    info_path = locate(root, match_id, 'matchinformation')
    position_path = locate(root, match_id, 'positions')
    meta = ET.parse(info_path).getroot()
    general = meta.find('.//General')
    if general is None or general.get('MatchId') != 'DFL-MAT-' + match_id:
        raise ValueError('Metadata match mismatch')
    team_ids = [general.get('HomeTeamId'), general.get('GuestTeamId')]
    if None in team_ids or len(set(team_ids)) != 2:
        raise ValueError('Invalid team mapping')
    groups = {}
    pitch = None
    stack = []
    current = None
    rows = []
    for event, e in ET.iterparse(position_path, events=('start', 'end')):
        if event == 'start':
            stack.append(e)
            if e.tag == 'FrameSet':
                current = dict(e.attrib)
                rows = []
                if current.get('MatchId') != general.get('MatchId'):
                    raise ValueError('Position match mismatch')
            continue
        if e.tag == 'PitchSize':
            pitch = (float(e.get('X')), float(e.get('Y')))
        elif e.tag == 'Frame':
            if current is None:
                raise ValueError('Frame outside FrameSet')
            rows.append((int(e.get('N')), float(e.get('X', 'nan')), float(e.get('Y', 'nan')), float(e.get('S', 'nan')), int(e.get('BallStatus', '-1')), int(e.get('BallPossession', '-1'))))
            stack[-2].remove(e)
        elif e.tag == 'FrameSet':
            half = current['GameSection']
            team = current['TeamId']
            if half not in ('firstHalf', 'secondHalf'):
                raise ValueError('Unsupported period')
            if team == 'BALL' or team in team_ids:
                a = np.asarray(rows, dtype=np.float64)
                if a.ndim != 2 or np.any(np.diff(a[:, 0]) <= 0):
                    raise ValueError('Duplicate or unordered frames')
                groups.setdefault(half, []).append((team, current['PersonId'], a))
            rows = []
            current = None
            stack[-2].remove(e)
        stack.pop()
    if pitch is None or set(groups) != {'firstHalf', 'secondHalf'}:
        raise ValueError('Incomplete match')
    halves = []
    for name in ('firstHalf', 'secondHalf'):
        ball_groups = [a for t, _, a in groups[name] if t == 'BALL']
        if len(ball_groups) != 1:
            raise ValueError('Expected one ball stream per half')
        b = ball_groups[0]
        frame_ids = b[:, 0].astype(np.int64)
        if np.any(np.diff(frame_ids) != 1):
            raise ValueError('Ball timeline has gaps; explicit reindexing needed')
        player_groups = [(t, pid, a) for t, pid, a in groups[name] if t != 'BALL']
        identities = sorted(set(((t, pid) for t, pid, _ in player_groups)))
        slots = {key: i for i, key in enumerate(identities)}
        pos = np.full((len(b), len(slots), 2), np.nan, np.float32)
        speeds = np.full(pos.shape[:2], np.nan, np.float32)
        occupied = np.zeros(pos.shape[:2], bool)
        for t, pid, a in player_groups:
            indices = a[:, 0].astype(np.int64) - frame_ids[0]
            j = slots[t, pid]
            if np.any(indices < 0) or np.any(indices >= len(b)):
                raise ValueError('Player outside ball timeline')
            if occupied[indices, j].any():
                raise ValueError('Overlapping identity records')
            occupied[indices, j] = True
            pos[indices, j] = a[:, 1:3] + np.asarray(pitch) / 2
            speeds[indices, j] = a[:, 3] / 3.6
        teams = np.array([team_ids.index(t) for t, _ in identities], dtype=np.int8)
        obs = Observations(frame_ids, (b[:, 1:3] + np.asarray(pitch) / 2).astype(np.float32), pos, speeds, teams, pitch)
        halves.append(Half(name, obs, labels(b[:, 4], b[:, 5])))
    return (halves, {p.name: sha256(p) for p in (info_path, position_path)})

def summarize(half):
    o = half.observed
    return dict(half=half.name, frames=len(o.ball), slots=len(o.teams), pitch_m=list(o.pitch), frame_contiguous=bool(np.all(np.diff(o.frame_ids) == 1)), ball_observed_fraction=float(np.isfinite(o.ball).all(axis=1).mean()), mean_players_observed=float(np.isfinite(o.players).all(axis=2).sum(axis=1).mean()), labels={c: int(np.sum(half.targets == c)) for c in 'HADX'})
