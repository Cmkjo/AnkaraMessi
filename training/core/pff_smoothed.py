"""Additional smoothed observations; duplicate/conflicting identities audited, no targets."""
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import pff_reader as pff

def load(folder, mid, parts, audit=None):
    root = Path(folder) / mid
    meta = json.loads((root / (mid + '.json')).read_text())[0]
    tids = [str(meta[k]['id']) for k in ('homeTeam', 'awayTeam')]
    roster = json.loads((root / f'Roster_{mid}.json').read_text())
    keys = sorted(((tids.index(str(p['team']['id'])), str(int(p['shirtNumber']))) for p in roster))
    slots = {k: i for i, k in enumerate(keys)}
    result = {}
    stats = dict(repeated_identity_entries=0, conflicting_player_frame_slots=0, conflicting_ball_frames=0)
    for p in parts:
        o = p.observed
        obs = replace(o, ball=np.full_like(o.ball, np.nan), players=np.full_like(o.players, np.nan), speeds=np.full_like(o.speeds, np.nan))
        result[p.number] = pff.Period(p.number, obs, p.reference, np.zeros_like(p.visible_players), np.zeros_like(p.visible_ball), p.present.copy())

    def flush(group):
        first = group[0]
        number = first['period']
        key = first['frameNum']
        if number not in result:
            return
        if any((r['period'] != number for r in group)):
            raise ValueError('Conflicting period')
        p = result[number]
        i = key - int(p.observed.frame_ids[0])
        values = {}
        balls = set()
        for row in group:
            for team, field in enumerate(('homePlayersSmoothed', 'awayPlayersSmoothed')):
                for player in row.get(field) or []:
                    k = (team, str(int(player['jerseyNum'])))
                    if k not in slots:
                        raise ValueError('Unknown smoothed identity')
                    value = (player.get('x'), player.get('y'), player.get('visibility'))
                    if k in values:
                        stats['repeated_identity_entries'] += 1
                    values.setdefault(k, set()).add(value)
            b = row.get('ballsSmoothed')
            if b:
                balls.add((b.get('x'), b.get('y'), b.get('visibility')))
        for k, v in values.items():
            if len(v) != 1:
                stats['conflicting_player_frame_slots'] += 1
                continue
            x, y, visibility = next(iter(v))
            j = slots[k]
            p.observed.players[i, j] = [np.nan if x is None else x, np.nan if y is None else y]
            p.visible_players[i, j] = visibility == 'VISIBLE'
        if len(balls) == 1:
            x, y, visibility = next(iter(balls))
            p.observed.ball[i] = [np.nan if x is None else x, np.nan if y is None else y]
            p.visible_ball[i] = visibility == 'VISIBLE'
        elif len(balls) > 1:
            stats['conflicting_ball_frames'] += 1
    group = []
    previous = None
    with (root / (mid + '.jsonl')).open() as stream:
        for line in stream:
            row = json.loads(line)
            key = row['frameNum']
            if previous is not None and key < previous:
                raise ValueError('Unordered frame')
            if previous is not None and key != previous:
                flush(group)
                group = []
            group.append(row)
            previous = key
    if group:
        flush(group)
    for p in result.values():
        p.observed.ball += np.array(p.observed.pitch) / 2
        p.observed.players += np.array(p.observed.pitch) / 2
        if np.isinf(p.observed.ball).any() or np.isinf(p.observed.players).any():
            raise ValueError('Infinite smoothed value')
    if audit is not None:
        audit.update(stats)
    return [result[k] for k in sorted(result)]
