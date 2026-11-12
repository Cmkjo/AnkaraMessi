"""PFF v4 raw observations and separate event-derived reference; streaming JSONL."""
from pathlib import Path
from dataclasses import dataclass, replace
import json, collections
import numpy as np
from dfl import Observations
import motion_model as mm

@dataclass
class Period:
    number: int
    observed: Observations
    reference: np.ndarray
    visible_players: np.ndarray
    visible_ball: np.ndarray
    present: np.ndarray

def label_step(event, state, owner):
    if not event:
        return (state, owner)
    kind = event.get('game_event_type')
    if kind in ('OUT', 'END'):
        state = 'D'
    elif kind in ('OTB', 'FIRSTKICKOFF', 'SECONDKICKOFF', 'THIRDKICKOFF', 'FOURTHKICKOFF'):
        state = 'L'
    elif kind in ('VID', 'FOUL'):
        state = None
        owner = None
    value = event.get('home_ball')
    if value is not None:
        if type(value) is not bool:
            raise ValueError('Ambiguous possession flag')
        owner = 'H' if value else 'A'
    return (state, owner)

def frames(raw):
    """Merge adjacent same-frame rows only when observation payload is identical.

    Event associations can multiply a tracking frame. Conflicting reference
    semantics become unknown; never choose a team by file order.
    """
    group = []
    last = None

    def merged(rows):
        fields = ('frameNum', 'period', 'videoTimeMs', 'periodElapsedTime', 'homePlayers', 'awayPlayers', 'balls')
        signatures = {json.dumps({k: r.get(k) for k in fields}, sort_keys=True) for r in rows}
        if len(signatures) != 1:
            raise ValueError('Duplicate frame with different observations')
        semantics = {(r['game_event'].get('game_event_type'), r['game_event'].get('home_ball')) for r in rows if r.get('game_event')}
        out = rows[0].copy()
        out['_extra_rows'] = len(rows) - 1
        out['_reference_conflict'] = len(semantics) > 1
        if len(semantics) == 1:
            kind, owner = next(iter(semantics))
            out['game_event'] = {'game_event_type': kind, 'home_ball': owner}
        elif len(semantics) > 1:
            out['game_event'] = None
        return out
    with open(raw) as stream:
        for line in stream:
            row = json.loads(line)
            key = row['frameNum']
            if last is not None and key < last:
                raise ValueError('Out-of-order frame')
            if last is not None and key != last:
                yield merged(group)
                group = []
            group.append(row)
            last = key
    if group:
        yield merged(group)

def load(folder, mid):
    folder = Path(folder) / mid
    meta = json.loads((folder / (mid + '.json')).read_text())[0]
    if str(meta['id']) != mid or abs(meta['fps'] - 29.97) > 1e-06:
        raise ValueError('Unsupported match/cadence')
    tids = [str(meta[k]['id']) for k in ('homeTeam', 'awayTeam')]
    pitch = meta['stadium']['pitches'][0]
    pitch = (float(pitch['length']), float(pitch['width']))
    roster = json.loads((folder / f'Roster_{mid}.json').read_text())
    identities = {}
    for p in roster:
        key = (tids.index(str(p['team']['id'])), str(int(p['shirtNumber'])))
        if key in identities:
            raise ValueError('Ambiguous roster jersey')
        identities[key] = str(p['player']['id'])
    keys = sorted(identities)
    slots = {k: i for i, k in enumerate(keys)}
    teams = np.array([k[0] for k in keys])
    raw = folder / (mid + '.jsonl')
    bounds = {}
    counts = collections.Counter()
    event_types = collections.Counter()
    first_keys = None
    previous = -1
    duplicates = 0
    conflicts = 0
    for row in frames(raw):
        frame = row['frameNum']
        period = row['period']
        duplicates += row['_extra_rows']
        conflicts += row['_reference_conflict']
        if frame <= previous:
            raise ValueError('Unordered/duplicate frame')
        previous = frame
        counts[str(period)] += 1
        if first_keys is None:
            first_keys = sorted((k for k in row if not k.startswith('_')))
        if period not in (1, 2):
            continue
        if period not in bounds:
            bounds[period] = [frame, frame]
        bounds[period][1] = frame
        if row.get('game_event'):
            event_types[row['game_event'].get('game_event_type')] += 1
    parts = {}
    for number, (start, end) in bounds.items():
        n = end - start + 1
        players = np.full((n, len(keys), 2), np.nan, dtype=np.float32)
        o = Observations(np.arange(start, end + 1), np.full((n, 2), np.nan, dtype=np.float32), players, np.full(players.shape[:2], np.nan, dtype=np.float32), teams, pitch, meta['fps'])
        parts[number] = Period(number, o, np.full(n, 'X'), np.zeros(players.shape[:2], bool), np.zeros(n, bool), np.zeros(n, bool))
    states = {k: (None, None) for k in parts}
    last_frame = {}
    max_clock_error = 0.0
    speed_values = 0
    multi_ball = 0
    for row in frames(raw):
        number = row['period']
        if number not in parts:
            continue
        p = parts[number]
        frame = row['frameNum']
        i = frame - bounds[number][0]
        p.present[i] = True
        if row.get('gameRefId') is not None and int(row['gameRefId']) != int(mid):
            raise ValueError('Tracking match mismatch')
        expected = (frame - bounds[number][0]) / meta['fps']
        if row['periodElapsedTime'] is not None:
            max_clock_error = max(max_clock_error, abs(float(row['periodElapsedTime']) - expected))
        seen = set()
        for team, field in enumerate(('homePlayers', 'awayPlayers')):
            for player in row[field] or []:
                key = (team, str(int(player['jerseyNum'])))
                if key not in slots or key in seen:
                    raise ValueError('Unknown/duplicate jersey')
                seen.add(key)
                j = slots[key]
                xy = [player.get('x'), player.get('y')]
                p.observed.players[i, j] = [np.nan if v is None else float(v) for v in xy]
                p.visible_players[i, j] = player.get('visibility') == 'VISIBLE'
                if player.get('speed') is not None:
                    speed_values += 1
        balls = row.get('balls') or []
        if len(balls) == 1:
            p.observed.ball[i] = [np.nan if balls[0].get(k) is None else float(balls[0][k]) for k in ('x', 'y')]
            p.visible_ball[i] = balls[0].get('visibility') == 'VISIBLE'
        elif len(balls) > 1:
            multi_ball += 1
        if number in last_frame and frame != last_frame[number] + 1:
            states[number] = (None, None)
        states[number] = (None, None) if row['_reference_conflict'] else label_step(row.get('game_event'), *states[number])
        last_frame[number] = frame
        state, owner = states[number]
        p.reference[i] = 'D' if state == 'D' else owner if state == 'L' and owner else 'X'
    if set(parts) != {1, 2}:
        raise ValueError('Missing half')
    for p in parts.values():
        p.observed.ball += np.array(pitch) / 2
        p.observed.players += np.array(pitch) / 2
        if np.isinf(p.observed.ball).any() or np.isinf(p.observed.players).any():
            raise ValueError('Infinite coordinate')
    return ([parts[k] for k in sorted(parts)], dict(match=mid, home=meta['homeTeam']['name'], away=meta['awayTeam']['name'], native_fps=meta['fps'], pitch_m=pitch, raw_keys=first_keys, rows_by_period=dict(counts), event_type_frame_counts=dict(event_types), player_slots=len(keys), max_elapsed_vs_frame_seconds=max_clock_error, multiple_ball_rows=multi_ball, duplicate_rows_merged=duplicates, reference_conflict_frames=conflicts, provider_speed_values_ignored=speed_values, reference_source='event-state reconstruction, NOT independent frame gold'))

def observations(period, scenario):
    if scenario not in ('all_finite', 'visible_only'):
        raise ValueError('Unknown quality policy')
    o = period.observed
    players = o.players.copy()
    ball = o.ball.copy()
    if scenario == 'visible_only':
        players[~period.visible_players] = np.nan
        ball[~period.visible_ball] = np.nan
    v, _ = mm.velocity(players, o.frame_ids, o.fps, 2)
    return replace(o, players=players, ball=ball, speeds=np.linalg.norm(v, axis=-1))

def grid_indices(n, fps, hz=5):
    grid = np.arange(int(np.floor((n - 1) / fps * hz)) + 1) / hz
    selected = np.floor(grid * fps + 0.5).astype(int)
    if np.any(np.diff(selected) <= 0):
        raise ValueError('Grid duplicates')
    hold = np.maximum(np.searchsorted(grid, np.arange(n) / fps, side='right') - 1, 0)
    return (selected, hold)
