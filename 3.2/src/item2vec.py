"""Session Item2Vec and candidate features, built only from observed histories."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from gensim.models import Word2Vec

ITEM_FEATURE_VERSION = 1
MILLISECONDS_PER_DAY = 86_400_000


@dataclass
class ProductHistory:
    products: np.ndarray
    timestamps: np.ndarray
    kinds: np.ndarray  # 0 other, 1 view, 2 click, 3 historical target-type click
    sessions: np.ndarray


@dataclass
class ItemHistories:
    example_ids: np.ndarray
    client_ids: np.ndarray
    last_timestamps: np.ndarray
    histories: list[ProductHistory]
    targets: np.ndarray | None


@dataclass
class ItemEmbedding:
    vectors: np.ndarray  # row = encoded product; unknown products have zero vectors

    def lookup(self, products):
        products = np.asarray(products, dtype=np.int64)
        result = np.zeros((len(products), self.vectors.shape[1]), dtype=np.float32)
        valid = (products >= 0) & (products < len(self.vectors))
        result[valid] = self.vectors[products[valid]]
        return result


def stable_hash(value):
    """Gensim initialization independent of Python's randomized hash seed."""
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:4], "little")


@lru_cache(maxsize=1)
def verify_gensim_blas():
    """Reject Gensim builds that replace a valid dot product of -1 with zero.

    Upstream fix: https://github.com/piskvorky/gensim/pull/3668
    Exercise the public scoring API instead of touching native pointers.
    """
    probe = Word2Vec(vector_size=4, min_count=1, workers=1, hs=1, negative=0, sg=1, window=1)
    probe.build_vocab([["left", "right"]])
    probe.wv.vectors[:] = 0
    probe.wv.vectors[:, 0] = 1
    probe.syn1[:] = 0
    probe.syn1[:, 0] = 1
    positive = probe.score([["left", "right"]], total_sentences=1)
    probe.syn1[:, 0] = -1
    negative = probe.score([["left", "right"]], total_sentences=1)
    if not np.allclose(positive, negative):
        raise RuntimeError("Gensim BLAS regression detected. Run 'uv sync --frozen' to install the pinned upstream fix.")
    return True


def load_item_histories(path, dictionary_path, cache_dir, max_events=2000, log=print):
    path, dictionary_path, cache_dir = Path(path), Path(dictionary_path), Path(cache_dir)
    stat = path.stat()
    signature = hashlib.sha256(dictionary_path.read_bytes()).hexdigest()
    key = hashlib.sha256(f"{ITEM_FEATURE_VERSION}|{stat.st_size}|{stat.st_mtime_ns}|{max_events}|{signature}".encode()).hexdigest()[:16]
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"items_{path.stem}_{key}.joblib"
    if cache_path.exists():
        log(f"Loading product-history cache: {path.name}")
        return joblib.load(cache_path)
    dictionary = pd.read_parquet(dictionary_path)
    kinds = {}
    boundaries = set()
    for row in dictionary.itertuples(index=False):
        kinds[row.encoded_event] = 3 if row.is_target_event else {"view": 1, "click": 2}.get(row.event_type, 0)
        if row.event_type == "session_boundary":
            boundaries.add(row.encoded_event)
    ids, clients, last_times, histories, targets = [], [], [], [], []
    parquet = pq.ParquetFile(path)
    has_target = "target_product" in parquet.schema_arrow.names
    for batch in parquet.iter_batches(batch_size=128):
        for row in batch.to_pylist():
            events = (row["events"] or [])[-max_events:]
            ids.append(row["example_id"])
            clients.append(row["client_id"])
            last_times.append(events[-1]["timestamp"] if events else 0)
            products, timestamps, types, sessions = [], [], [], []
            previous_time, session = None, 0
            for event in events:
                timestamp = event["timestamp"]
                if event["encoded_event"] in boundaries or (previous_time is not None and timestamp - previous_time > 30 * 60_000):
                    session += 1
                previous_time = timestamp
                if event["encoded_product"] is None:
                    continue
                products.append(event["encoded_product"])
                timestamps.append(timestamp)
                types.append(kinds.get(event["encoded_event"], 0))
                sessions.append(session)
            histories.append(ProductHistory(np.asarray(products, dtype=np.int32), np.asarray(timestamps, dtype=np.int64),
                                            np.asarray(types, dtype=np.uint8), np.asarray(sessions, dtype=np.int32)))
            if has_target:
                targets.append(row["target_product"])
        if len(ids) % 10240 == 0:
            log(f"{path.name}: product histories {len(ids):,}/{parquet.metadata.num_rows:,}")
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate example IDs")
    result = ItemHistories(np.asarray(ids), np.asarray(clients), np.asarray(last_times, dtype=np.int64), histories,
                           np.asarray(targets, dtype=np.int32) if has_target else None)
    joblib.dump(result, cache_path, compress=3)
    log(f"Saved product-history cache: {cache_path.name}")
    return result


class SessionCorpus:
    def __init__(self, data, indices):
        self.data, self.indices = data, np.asarray(indices)

    def __iter__(self):
        tokens = {}
        for index in self.indices:
            history = self.data.histories[index]
            session_tokens, previous_session, previous_product = [], None, None
            for product, session in zip(history.products, history.sessions):
                product, session = int(product), int(session)
                if session != previous_session:
                    if len(session_tokens) >= 2:
                        yield session_tokens
                    session_tokens, previous_product = [], None
                if product != previous_product:
                    if product not in tokens:
                        tokens[product] = f"p{product}"
                    session_tokens.append(tokens[product])
                previous_product, previous_session = product, session
            if len(session_tokens) >= 2:
                yield session_tokens


def train_embedding(data, indices, classes, dimensions=24, epochs=8, seed=42, log=print):
    verify_gensim_blas()
    corpus = SessionCorpus(data, indices)
    model = Word2Vec(vector_size=dimensions, window=5, min_count=1, sg=1, negative=5,
                     sample=0, workers=1, seed=seed, hashfxn=stable_hash, sorted_vocab=1)
    model.build_vocab(corpus)
    if not model.corpus_count:
        raise ValueError("No multi-product sessions for Item2Vec")
    log(f"Item2Vec: {model.corpus_count:,} sessions; {model.corpus_total_words:,} words; {len(model.wv)} products; epochs={epochs}; workers=1")
    model.train(corpus, total_examples=model.corpus_count, epochs=epochs)
    maximum = max(int(max(classes)), max(int(token[1:]) for token in model.wv.index_to_key))
    vectors = np.zeros((maximum + 1, dimensions), dtype=np.float32)
    for token in model.wv.index_to_key:
        vector = model.wv[token]
        vectors[int(token[1:])] = vector / max(float(np.linalg.norm(vector)), 1e-12)
    return ItemEmbedding(vectors), model


def normalized_profile(vectors, weights=None):
    if not len(vectors):
        return np.zeros(vectors.shape[1], dtype=np.float32)
    mean = vectors.mean(axis=0) if weights is None else np.average(vectors, axis=0, weights=weights)
    return (mean / max(float(np.linalg.norm(mean)), 1e-12)).astype(np.float32)


def candidate_features(data, indices, classes, embedding, log=print):
    """Score ALL classes: no retrieval truncation and no use of target labels."""
    indices, classes = np.asarray(indices), np.asarray(classes, dtype=np.int32)
    n_classes, dimensions = len(classes), embedding.vectors.shape[1]
    names = ["candidate_product", "cosine_mean", "cosine_decay", "cosine_clicks", "cosine_recent10", "cosine_last",
             "log_count", "product_share", "log_views", "log_clicks", "log_target_clicks", "decayed_share",
             "recent20_share", "recent100_share", "last_session_share", "log_recency_days", "position_since_last",
             "is_last_product", "is_last_click", "is_last_target_click", "was_seen", "log_history_length"]
    names.extend(f"item_vector_{i}" for i in range(dimensions))
    matrix = np.zeros((len(indices) * n_classes, len(names)), dtype=np.float32)
    user_features = np.zeros((len(indices), dimensions * 4 + n_classes * 7), dtype=np.float32)
    item_vectors = embedding.lookup(classes)
    cosine_scores = np.zeros((len(indices), n_classes), dtype=np.float32)
    for row_number, index in enumerate(indices):
        history = data.histories[index]
        products, times, kinds, sessions = history.products, history.timestamps, history.kinds, history.sessions
        block = matrix[row_number * n_classes:(row_number + 1) * n_classes]
        block[:, 0], block[:, 22:] = classes, item_vectors
        if not len(products):
            block[:, 15:17] = 5.0, 1.0
            continue
        size = max(int(products.max()), int(classes.max())) + 1
        age_days = np.maximum(0, data.last_timestamps[index] - times) / MILLISECONDS_PER_DAY
        weights = np.exp2(-age_days / 3).astype(np.float32)
        counts = np.bincount(products, minlength=size).astype(np.float32)
        views = np.bincount(products[kinds == 1], minlength=size).astype(np.float32)
        clicks = np.bincount(products[kinds >= 2], minlength=size).astype(np.float32)
        targets = np.bincount(products[kinds == 3], minlength=size).astype(np.float32)
        decay = np.bincount(products, weights=weights, minlength=size).astype(np.float32)
        recent20 = np.bincount(products[-20:], minlength=size).astype(np.float32)
        recent100 = np.bincount(products[-100:], minlength=size).astype(np.float32)
        in_session = np.bincount(products[sessions == sessions[-1]], minlength=size).astype(np.float32)
        last_positions = np.full(size, -1, dtype=np.int32)
        np.maximum.at(last_positions, products, np.arange(len(products)))
        positions = last_positions[classes]
        recency = np.full(n_classes, 5.0, dtype=np.float32)
        seen = positions >= 0
        recency[seen] = np.log1p(np.maximum(0, data.last_timestamps[index] - times[positions[seen]]) / MILLISECONDS_PER_DAY)
        vectors = embedding.lookup(products)
        profiles = np.stack([
            normalized_profile(vectors), normalized_profile(vectors, weights),
            normalized_profile(vectors[kinds >= 2]), normalized_profile(vectors[-10:]),
        ])
        cosines = profiles @ item_vectors.T
        cosine_scores[row_number] = cosines[1]
        block[:, 1:5] = cosines.T
        block[:, 5] = vectors[-1] @ item_vectors.T
        block[:, 6] = np.log1p(counts[classes])
        block[:, 7] = counts[classes] / len(products)
        block[:, 8:11] = np.stack([np.log1p(views[classes]), np.log1p(clicks[classes]), np.log1p(targets[classes])], axis=1)
        block[:, 11] = decay[classes] / max(float(decay.sum()), 1e-12)
        block[:, 12] = recent20[classes] / min(20, len(products))
        block[:, 13] = recent100[classes] / min(100, len(products))
        block[:, 14] = in_session[classes] / max(1, int(in_session.sum()))
        block[:, 15] = recency
        block[:, 16] = np.where(seen, (len(products) - 1 - positions) / len(products), 1)
        block[:, 17] = classes == products[-1]
        block[:, 18] = classes == (products[kinds >= 2][-1] if np.any(kinds >= 2) else -1)
        block[:, 19] = classes == (products[kinds == 3][-1] if np.any(kinds == 3) else -1)
        block[:, 20] = seen
        block[:, 21] = np.log1p(len(products))
        user_features[row_number] = np.concatenate([profiles.ravel(), block[:, [1, 2, 3, 7, 10, 11, 15]].T.ravel()])
        if (row_number + 1) % 20000 == 0:
            log(f"Candidate features: {row_number + 1:,}/{len(indices):,} histories; {n_classes} candidates each")
    if not np.isfinite(matrix).all() or not np.isfinite(user_features).all():
        raise ValueError("Non-finite Item2Vec features")
    frame = pd.DataFrame(matrix, columns=names)
    frame["candidate_product"] = frame["candidate_product"].astype(np.int32)
    return frame, user_features, cosine_scores


def candidate_labels(targets, classes):
    targets, classes = np.asarray(targets), np.asarray(classes)
    if not np.isin(targets, classes).all() or len(np.unique(classes)) != len(classes):
        raise ValueError("Invalid candidate catalogue or target")
    labels = (targets[:, None] == classes[None, :]).astype(np.int8)
    if not np.all(labels.sum(axis=1) == 1):
        raise ValueError("Exactly one positive candidate is required per history")
    return labels.ravel(), np.repeat(np.arange(len(targets), dtype=np.int64), len(classes))
