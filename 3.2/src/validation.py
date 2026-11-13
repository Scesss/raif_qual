"""Chronological validation using the final observed event, not a hidden click."""

import numpy as np


def chronological_split(last_timestamps, client_ids, validation_size=0.2):
    timestamps = np.asarray(last_timestamps)
    if not 0 < validation_size < 1 or len(timestamps) < 2:
        raise ValueError("Invalid temporal validation split")
    if len(np.unique(client_ids)) != len(client_ids):
        raise ValueError("Chronological split expects one history per client")
    boundary = min(len(timestamps) - 1, max(1, int(len(timestamps) * (1 - validation_size))))
    cutoff = np.sort(timestamps)[boundary]
    fit = np.flatnonzero(timestamps < cutoff)
    validation = np.flatnonzero(timestamps >= cutoff)
    if not len(fit) or not len(validation):
        raise ValueError("No strict time boundary: all timestamps may be identical")
    return fit, validation, int(cutoff)
