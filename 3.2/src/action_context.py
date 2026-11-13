"""Observed action context and fit-only relations; no hidden labels are read."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.features import file_sha256

DAY = 86_400_000


@dataclass
class RawHistory:
    events: np.ndarray
    products: np.ndarray
    times: np.ndarray


@dataclass
class RawData:
    example_ids: np.ndarray
    histories: list[RawHistory]


def load_raw(path, cache_dir, max_events=256, log=print):
    path, cache_dir = Path(path), Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    signature = {"source_sha256": file_sha256(path), "max_events": max_events, "version": 1}
    cache = cache_dir / f"raw_{signature['source_sha256'][:12]}_{max_events}.joblib"
    if cache.exists():
        log("Loading cached raw action tails")
        return joblib.load(cache)
    ids, histories = [], []
    for batch in pq.ParquetFile(path).iter_batches(batch_size=128, columns=["example_id", "events"]):
        for row in batch.to_pylist():
            events = (row["events"] or [])[-max_events:]
            ids.append(row["example_id"])
            histories.append(RawHistory(
                np.asarray([e["encoded_event"] for e in events], dtype=np.int16),
                np.asarray([-1 if e["encoded_product"] is None else e["encoded_product"] for e in events], dtype=np.int16),
                np.asarray([e["timestamp"] for e in events], dtype=np.int64)))
        if len(ids) % 20480 == 0:
            log(f"Raw actions: {len(ids):,} histories")
    result = RawData(np.asarray(ids), histories)
    joblib.dump(result, cache, compress=3)
    return result


def fit_context_schema(raw, indices, event_count, top_k=8):
    counts = np.zeros(event_count, dtype=np.int64)
    pairs = Counter()
    for i in indices:
        h = raw.histories[i]
        counts += np.bincount(h.events[h.products >= 0], minlength=event_count)
        for lag in (0, 1):
            if len(h.events) >= lag + 2:
                pairs[tuple(map(int, h.events[len(h.events)-lag-2:len(h.events)-lag]))] += 1
    top = np.argsort(-counts, kind="stable")[:top_k]
    top = top[counts[top] > 0]
    vocabulary = {p: k + 1 for k, (p, _) in enumerate(pairs.most_common(512))}
    return {"top_action_codes": top.tolist(), "pair_vocabulary": vocabulary}


def action_features(raw, indices, classes, schema, kinds, boundaries, log=None):
    classes = np.asarray(classes)
    size = int(classes.max()) + 1
    numeric_names = ["raw_log_length", "raw_nonproduct_share", "raw_log_unique_actions", "raw_log_sessions",
                     "raw_last_gap", "raw_recent_median_gap", "raw_last_session_length", "raw_recent_product_diversity",
                     "candidate_raw_position", "candidate_log_unique_actions", "candidate_view_recency",
                     "candidate_click_recency", "candidate_target_recency", "candidate_click_after_view",
                     "candidate_view_click_gap", "candidate_recent8_share", "candidate_recent32_share",
                     "candidate_session_count", "candidate_raw_action_share"]
    numeric_names += [f"candidate_action_{code}_count" for code in schema["top_action_codes"]]
    cat_names = [f"raw_lag_{lag}_code" for lag in (1, 2, 3, 5, 10)] + ["raw_pair_last", "raw_pair_previous", "candidate_last_action"]
    matrix = np.zeros((len(indices) * len(classes), len(numeric_names) + len(cat_names)), dtype=np.float32)
    category_start = len(numeric_names)
    for row, i in enumerate(indices):
        h = raw.histories[i]
        block = matrix[row * len(classes):(row + 1) * len(classes)]
        e, p, t = h.events, h.products, h.times
        if not len(e):
            block[:, 8:13] = [1, 0, 5, 5, 5]
            continue
        event_kinds = kinds[e]
        session_changes = np.isin(e, list(boundaries)) | np.r_[False, np.diff(t) > 30 * 60_000]
        sessions = np.cumsum(session_changes)
        latest_session = sessions == sessions[-1]
        gaps = np.maximum(0, np.diff(t)) / 1000
        block[:, :8] = [np.log1p(len(e)), np.mean(p < 0), np.log1p(len(np.unique(e))),
                        np.log1p(int(session_changes.sum()) + 1), np.log1p(gaps[-1]) if len(gaps) else 0,
                        np.log1p(np.median(gaps[-10:])) if len(gaps) else 0, np.log1p(latest_session.sum()),
                        len(np.unique(p[-20:][p[-20:] >= 0])) / 20]
        for pos, lag in enumerate((1, 2, 3, 5, 10)):
            block[:, category_start + pos] = int(e[-lag]) if len(e) >= lag else 0
        for offset in (0, 1):
            pair = tuple(map(int, e[len(e)-offset-2:len(e)-offset])) if len(e) >= offset + 2 else ()
            block[:, category_start + 5 + offset] = schema["pair_vocabulary"].get(pair, 0)
        valid = (p >= 0) & (p < size)
        all_counts = np.bincount(p[valid], minlength=size)
        recent8 = np.bincount(p[-8:][(p[-8:] >= 0) & (p[-8:] < size)], minlength=size)
        recent32 = np.bincount(p[-32:][(p[-32:] >= 0) & (p[-32:] < size)], minlength=size)
        in_session = np.bincount(p[valid & latest_session], minlength=size)
        block[:, 15:19] = np.stack([recent8[classes] / min(8, len(e)), recent32[classes] / min(32, len(e)),
                                    np.log1p(in_session[classes]), all_counts[classes] / len(e)], axis=1)
        for col, code in enumerate(schema["top_action_codes"], start=19):
            counts = np.bincount(p[valid & (e == code)], minlength=size)
            block[:, col] = np.log1p(counts[classes])
        for j, candidate in enumerate(classes):
            positions = np.flatnonzero(p == candidate)
            block[j, 8] = (len(e) - 1 - positions[-1]) / len(e) if len(positions) else 1
            block[j, 9] = np.log1p(len(np.unique(e[positions])))
            if len(positions):
                block[j, category_start + 7] = e[positions[-1]]
            last_by_kind = []
            for col, mask in ((10, event_kinds == 1), (11, event_kinds >= 2), (12, event_kinds == 3)):
                found = positions[mask[positions]]
                last = int(found[-1]) if len(found) else -1
                last_by_kind.append(last)
                block[j, col] = np.log1p((t[-1] - t[last]) / DAY) if last >= 0 else 5
            view, click = last_by_kind[:2]
            block[j, 13] = view >= 0 and click > view
            block[j, 14] = np.log1p(abs(t[click] - t[view]) / 1000) if min(view, click) >= 0 else 0
        if log and (row + 1) % 20000 == 0:
            log(f"Raw context features: {row+1:,}/{len(indices):,} histories")
    frame = pd.DataFrame(matrix, columns=numeric_names + cat_names)
    for c in cat_names:
        frame[c] = frame[c].astype(np.int32)
    if not np.isfinite(matrix).all():
        raise ValueError("Non-finite raw action features")
    return frame, cat_names


def fit_relations(items, raw, indices, classes, event_count, target_codes):
    """Use only observed histories: neither supplied target_product nor held-out rows."""
    size = int(max(classes)) + 1
    step1 = np.zeros((size, size), dtype=np.float64)
    step2 = np.zeros_like(step1)
    covisit = np.zeros_like(step1)
    action_target = np.zeros((event_count, size), dtype=np.float64)
    historical_prior = np.zeros(size, dtype=np.float64)
    auxiliary_n = 0
    for i in indices:
        h = items.histories[i]
        p, s = h.products, h.sessions
        valid = (p >= 0) & (p < size)
        p, s = p[valid], s[valid]
        keep = np.r_[True, (p[1:] != p[:-1]) | (s[1:] != s[:-1])] if len(p) else np.array([], dtype=bool)
        p, s = p[keep], s[keep]
        for jump, dest in ((1, step1), (2, step2)):
            if len(p) > jump:
                take = (s[jump:] == s[:-jump]) & (p[jump:] != p[:-jump])
                np.add.at(dest, (p[:-jump][take], p[jump:][take]), 1)
        unique = np.unique(p[-64:])
        if len(unique):
            covisit[np.ix_(unique, unique)] += 1 / max(1, len(unique) - 1)
        r = raw.histories[i]
        for j in np.flatnonzero(np.isin(r.events, list(target_codes))):
            c = int(r.products[j])
            if j < 25 or c not in classes or np.any(np.isin(r.events[j-5:j], list(target_codes))):
                continue
            prefix = r.events[max(0, j-5-32):j-5]
            action_target[np.unique(prefix), c] += 1
            historical_prior[c] += 1
            auxiliary_n += 1
    destination_prior = (step1.sum(axis=0) + 1) / (step1.sum() + size)
    transition = (step1 + .5 * destination_prior) / (step1.sum(axis=1, keepdims=True) + .5)
    two_step = (step2 + .5 * destination_prior) / (step2.sum(axis=1, keepdims=True) + .5)
    lift = np.log1p(transition / destination_prior)
    diag = np.maximum(np.diag(covisit), 1e-9)
    covisit /= np.sqrt(diag[:, None] * diag[None, :])
    np.fill_diagonal(covisit, 0)
    hp = (historical_prior + 1) / (historical_prior.sum() + size)
    event_probability = (action_target + 5 * hp) / (action_target.sum(axis=1, keepdims=True) + 5)
    event_lift = np.clip(np.log(event_probability / hp), -5, 5)
    return {"transition": transition.astype(np.float32), "two_step": two_step.astype(np.float32),
            "lift": lift.astype(np.float32), "covisit": covisit.astype(np.float32),
            "event_probability": event_probability.astype(np.float32), "event_lift": event_lift.astype(np.float32),
            "auxiliary_n": auxiliary_n, "historical_target_counts": historical_prior.astype(int).tolist()}


def relation_features(items, raw, indices, classes, tables):
    names = ["transition_last", "transition_last_click", "transition_last_target", "transition_recent10_mean",
             "transition_recent10_max", "transition_recent10_lift", "transition_two_step_last",
             "covisit_last", "covisit_recent20_mean", "covisit_recent20_max", "event_target_last_lift",
             "event_target_recent8_lift", "event_target_recent32_lift", "event_target_recent32_max_lift",
             "event_target_recent32_probability"]
    matrix = np.zeros((len(indices) * len(classes), len(names)), dtype=np.float32)
    size = len(tables["transition"])
    for row, i in enumerate(indices):
        h = items.histories[i]
        block = matrix[row*len(classes):(row+1)*len(classes)]
        p = h.products[(h.products >= 0) & (h.products < size)]
        if len(p):
            block[:, 0] = tables["transition"][p[-1], classes]
            for col, mask in ((1, h.kinds >= 2), (2, h.kinds == 3)):
                prev = h.products[mask & (h.products >= 0) & (h.products < size)]
                if len(prev):
                    block[:, col] = tables["transition"][prev[-1], classes]
            recent = tables["transition"][np.ix_(p[-10:], classes)]
            block[:, 3:6] = np.stack([recent.mean(axis=0), recent.max(axis=0),
                                     tables["lift"][np.ix_(p[-10:], classes)].mean(axis=0)], axis=1)
            block[:, 6] = tables["two_step"][p[-1], classes]
            co = tables["covisit"][np.ix_(p[-20:], classes)]
            block[:, 7:10] = np.stack([co[-1], co.mean(axis=0), co.max(axis=0)], axis=1)
        e = raw.histories[i].events
        if len(e):
            lift = tables["event_lift"][np.ix_(e[-32:], classes)]
            block[:, 10:15] = np.stack([lift[-1], lift[-8:].mean(axis=0), lift.mean(axis=0), lift.max(axis=0),
                                        tables["event_probability"][np.ix_(e[-32:], classes)].mean(axis=0)], axis=1)
    if not np.isfinite(matrix).all():
        raise ValueError("Non-finite relation features")
    return pd.DataFrame(matrix, columns=names)


def text_document(history, limit=128):
    e, p = history.events[-limit:], history.products[-limit:]
    tokens = []
    for j, (action, product) in enumerate(zip(e, p)):
        tokens.append(f"e{action}")
        if product >= 0:
            tokens.extend((f"p{product}", f"j{action}_{product}"))
        if j >= len(e) - 32:
            tokens.append(f"recent_e{action}")
            if product >= 0:
                tokens.extend((f"recent_p{product}", f"recent_j{action}_{product}"))
            if j:
                tokens.append(f"b{e[j-1]}_{p[j-1]}_{action}_{product}")
    for lag, action in enumerate(e[::-1][:5]):
        tokens.append(f"lag{lag}_e{action}")
    return " ".join(tokens) or "empty_history"


def chronological_oof_plan(timestamps, fit_indices, warmup=20000, block_size=20000):
    fit_indices = np.asarray(fit_indices)
    ordered = fit_indices[np.argsort(np.asarray(timestamps)[fit_indices], kind="stable")]
    t = np.asarray(timestamps)
    plan = []
    for start in range(warmup, len(ordered), block_size):
        cutoff = t[ordered[start]]
        end = min(start + block_size, len(ordered))
        end_time = t[ordered[end]] if end < len(ordered) else t[ordered[-1]] + 1
        training = fit_indices[t[fit_indices] < cutoff]
        prediction = fit_indices[(t[fit_indices] >= cutoff) & (t[fit_indices] < end_time)]
        if len(training) and len(prediction):
            plan.append((training, prediction))
    return plan
