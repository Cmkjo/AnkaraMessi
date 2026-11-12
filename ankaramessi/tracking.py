"""Tracking observations and input integrity."""
from dataclasses import dataclass
import hashlib
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

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()
