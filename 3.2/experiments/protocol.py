"""Original chronological client roles and strict out-of-fold alignment."""

import numpy as np


def rolling_blocks(timestamps, client_ids, fraction):
    times = np.asarray(timestamps)
    if len(np.unique(client_ids)) != len(times):
        raise ValueError("Expected one original example per client")
    ordered = np.sort(times)
    cuts = [
        int(ordered[min(len(times) - 1, round(len(times) * q))])
        for q in (fraction, fraction + 0.05, fraction + 0.10, fraction + 0.20)
    ]
    if fraction + 0.20 >= 1 - 1e-9:
        cuts[-1] = int(times.max()) + 1
    groups = [np.flatnonzero(times < cuts[0])]
    groups.extend(
        np.flatnonzero((times >= a) & (times < b)) for a, b in zip(cuts[:-1], cuts[1:])
    )
    if any(not len(g) for g in groups):
        raise ValueError("Empty temporal block")
    for before, after in zip(groups[:-1], groups[1:]):
        if times[before].max() >= times[after].min():
            raise ValueError("Temporal overlap")
    return dict(zip(("fit", "early_stop", "calibration", "check"), groups))


def split_roles(store):
    reference = rolling_blocks(store.index["times"], store.index["clients"], 0.8)
    fit = reference["fit"]
    ordered = fit[np.argsort(store.index["times"][fit], kind="stable")]
    boundary = int(len(ordered) * 0.875)
    result = dict(
        gru_fit=ordered[:boundary],
        gbm_fit=ordered[boundary:],
        early_stop=reference["early_stop"],
        calibration=reference["calibration"],
        check=reference["check"],
    )
    roles = list(result.values())
    if len(np.unique(np.concatenate(roles))) != store.n_original:
        raise ValueError("Roles overlap or do not cover original clients")
    for before, after in zip(roles[:-1], roles[1:]):
        if store.index["times"][before].max() > store.index["times"][after].min():
            raise ValueError("Roles are not chronological")
    result["full_gru"] = np.concatenate(roles[:-1])
    result["full_gbm"] = result["check"].copy()
    if np.intersect1d(result["full_gru"], result["full_gbm"]).size:
        raise ValueError("Full GRU and GBM clients overlap")
    return result
