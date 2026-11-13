"""Turn a nested event history into tokens without using IDs or hidden labels."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pyarrow.parquet as pq

FEATURE_VERSION = 2
CALENDAR_TIMEZONE = "Europe/Moscow"


@dataclass
class Histories:
    example_ids: np.ndarray
    client_ids: np.ndarray
    texts: list[str]
    lengths: np.ndarray
    last_timestamps: np.ndarray
    last_target_products: np.ndarray
    targets: np.ndarray | None
    sequence_texts: list[str] | None = None
    decay_counts: list[dict[str, float]] | None = None
    calendar_texts: list[str] | None = None
    contexts: list[dict[str, str | float]] | None = None


@lru_cache(maxsize=16384)
def calendar_at_hour(unix_hour: int, timezone_name: str = CALENDAR_TIMEZONE) -> tuple[int, int]:
    """Moscow has whole-hour offsets; all events in a UTC hour share a date/hour."""
    local = datetime.fromtimestamp(unix_hour * 3600, tz=ZoneInfo(timezone_name))
    return local.weekday(), local.hour


def day_context(timestamp_ms: int, timezone_name: str = CALENDAR_TIMEZONE) -> tuple[str, int, int]:
    weekday, hour = calendar_at_hour(int(timestamp_ms) // 3_600_000, timezone_name)
    return ("weekend" if weekday >= 5 else "weekday"), weekday, hour


def behavior_features(
    events: list[dict],
    click_event_codes: set[int],
    max_events: int = 512,
    half_life_days: float = 3.0,
    timezone_name: str = CALENDAR_TIMEZONE,
) -> tuple[str, dict[str, float], str, dict[str, str | float]]:
    """Order, time decay, calendar interactions, and observed exploration proxies.

    Time is anchored at the last OBSERVED event; the hidden target's timestamp
    is unavailable. Novelty means first occurrence inside this history window.
    """
    if half_life_days <= 0:
        raise ValueError("Time-decay half life must be positive")
    recent = events[-max_events:]
    if not recent:
        return "empty_history", {"empty_history": 1.0}, "empty_history", {"day_type": "unknown"}
    reference_ms = recent[-1]["timestamp"]
    current_kind, weekday, hour = day_context(reference_ms, timezone_name)
    sequence, calendar_tokens = [], []
    weighted: dict[str, float] = {}
    context: dict[str, str | float] = {
        "day_type": current_kind,
        "weekday": str(weekday),
        "hour": str(hour),
        "hour_block": str(hour // 4),
    }
    seen_products: set[int] = set()
    by_kind = {
        kind: {"events": 0, "product_events": 0, "novel": 0, "products": set()}
        for kind in ["weekday", "weekend"]
    }
    previous_product = last_click_product = None
    for position, event in enumerate(recent):
        action, product = event["encoded_event"], event["encoded_product"]
        kind, _, _ = day_context(event["timestamp"], timezone_name)
        stats = by_kind[kind]
        stats["events"] += 1
        # One token per actual event: n-grams therefore represent event order.
        sequence.append(f"e{action}_p{product if product is not None else 'none'}")
        event_tokens = [f"e{action}"]
        if product is not None:
            event_tokens.extend([f"p{product}", f"ep{action}_{product}"])
            stats["product_events"] += 1
            stats["products"].add(product)
            stats["novel"] += product not in seen_products
            seen_products.add(product)
            previous_product = product
            if action in click_event_codes:
                last_click_product = product
            # Different product preferences depending on the latest day type.
            calendar_tokens.append(f"context_{current_kind}_p{product}")
        age_days = max(0, reference_ms - event["timestamp"]) / 86_400_000
        weight = math.exp(-math.log(2) * age_days / half_life_days)
        for token in event_tokens:
            weighted[token] = weighted.get(token, 0.0) + weight
            calendar_tokens.append(f"on_{kind}_{token}")
            if position >= len(recent) - 32:
                calendar_tokens.append(f"recent_{kind}_{token}")
    context["last_product"] = str(previous_product) if previous_product is not None else "none"
    context["last_click_product"] = str(last_click_product) if last_click_product is not None else "none"
    context["history_log_length"] = math.log1p(len(recent)) / math.log1p(max_events)
    for kind, stats in by_kind.items():
        product_events = stats["product_events"]
        context[f"{kind}_activity_share"] = stats["events"] / len(recent)
        context[f"{kind}_has_product"] = str(bool(product_events))
        context[f"{kind}_novelty_share"] = stats["novel"] / max(1, product_events)
        context[f"{kind}_diversity_share"] = len(stats["products"]) / max(1, product_events)
    return " ".join(sequence), weighted, " ".join(calendar_tokens), context


def model_inputs(histories: Histories, feature_set: str, indices=None) -> list:
    selected = range(len(histories.texts)) if indices is None else indices
    if feature_set == "baseline":
        return [histories.texts[i] for i in selected]
    if any(value is None for value in [histories.sequence_texts, histories.decay_counts, histories.calendar_texts, histories.contexts]):
        raise ValueError("Behavior features are missing; recreate the feature cache")
    return [
        {"text": histories.texts[i], "sequence": histories.sequence_texts[i],
         "decay": histories.decay_counts[i], "calendar": histories.calendar_texts[i],
         "context": histories.contexts[i]}
        for i in selected
    ]


def history_to_text(events: list[dict], max_events: int = 512) -> str:
    """Bag of events/products plus a separate vocabulary for recent actions.

    The latest 512 events limit runtime and memory. Positional tokens retain
    a little ordering information; TF-IDF uses repeated tokens as counts.
    """
    if not events:
        return "empty_history"
    recent = events[-max_events:]
    tokens: list[str] = []
    for event in recent:
        action = event["encoded_event"]
        product = event["encoded_product"]
        tokens.append(f"e{action}")
        if product is not None:
            tokens.extend((f"p{product}", f"ep{action}_{product}"))
    for event in recent[-32:]:
        action = event["encoded_event"]
        product = event["encoded_product"]
        tokens.append(f"recent_e{action}")
        if product is not None:
            tokens.extend((f"recent_p{product}", f"recent_ep{action}_{product}"))
    for lag, event in enumerate(reversed(recent[-5:])):
        tokens.append(f"lag{lag}_e{event['encoded_event']}")
        if event["encoded_product"] is not None:
            tokens.append(f"lag{lag}_p{event['encoded_product']}")
    return " ".join(tokens)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_histories(
    path: Path,
    cache_dir: Path,
    target_event_codes: set[int],
    max_events: int = 512,
    log: Callable[[str], None] = print,
    feature_set: str = "baseline",
    click_event_codes: set[int] | None = None,
    half_life_days: float = 3.0,
    timezone_name: str = CALENDAR_TIMEZONE,
) -> Histories:
    """Read in small batches instead of materializing all nested Python rows."""
    stat = path.stat()
    if feature_set not in {"baseline", "behavior"}:
        raise ValueError(f"Unknown feature set: {feature_set}")
    version = 1 if feature_set == "baseline" else FEATURE_VERSION
    key = f"{version}|{stat.st_size}|{stat.st_mtime_ns}|{max_events}|{sorted(target_event_codes)}"
    if feature_set == "behavior":
        key += f"|{sorted(click_event_codes or set())}|{half_life_days}|{timezone_name}"
    cache_key = hashlib.sha256(key.encode()).hexdigest()[:16]
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{path.stem}_{cache_key}.joblib"
    if cache_path.exists():
        log(f"Loading cached features: {path.name}")
        return joblib.load(cache_path)

    parquet = pq.ParquetFile(path)
    ids, clients, texts, lengths, times, last_products, targets = [], [], [], [], [], [], []
    sequences, decays, calendars, contexts = [], [], [], []
    has_target = "target_product" in parquet.schema_arrow.names
    for batch in parquet.iter_batches(batch_size=128):
        for row in batch.to_pylist():
            events = row["events"] or []
            ids.append(row["example_id"])
            clients.append(row["client_id"])
            texts.append(history_to_text(events, max_events))
            if feature_set == "behavior":
                sequence, decay, calendar, context = behavior_features(
                    events, click_event_codes or set(), max_events, half_life_days, timezone_name
                )
                sequences.append(sequence)
                decays.append(decay)
                calendars.append(calendar)
                contexts.append(context)
            lengths.append(len(events))
            times.append(events[-1]["timestamp"] if events else 0)
            previous_product = next(
                (
                    event["encoded_product"]
                    for event in reversed(events)
                    if event["encoded_event"] in target_event_codes
                    and event["encoded_product"] is not None
                ),
                -1,
            )
            last_products.append(previous_product)
            if has_target:
                targets.append(row["target_product"])
        if len(ids) % 5120 == 0:
            log(f"{path.name}: {len(ids):,}/{parquet.metadata.num_rows:,} histories")

    result = Histories(
        example_ids=np.asarray(ids),
        client_ids=np.asarray(clients),
        texts=texts,
        lengths=np.asarray(lengths, dtype=np.int32),
        last_timestamps=np.asarray(times, dtype=np.int64),
        last_target_products=np.asarray(last_products, dtype=np.int32),
        targets=np.asarray(targets, dtype=np.int32) if has_target else None,
        sequence_texts=sequences if feature_set == "behavior" else None,
        decay_counts=decays if feature_set == "behavior" else None,
        calendar_texts=calendars if feature_set == "behavior" else None,
        contexts=contexts if feature_set == "behavior" else None,
    )
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate example_id in {path}")
    joblib.dump(result, cache_path, compress=3)
    log(f"Saved feature cache: {cache_path.name}")
    return result
