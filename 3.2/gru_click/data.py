"""Causal prefix inputs. Every client belongs to one role, including past clicks."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from .common import atomic_json, sha256

TOKENS = ("events", "products", "gaps", "hours", "weekdays", "boundaries")
FEATURES = (
    "log_count",
    "log_views",
    "log_clicks",
    "log_targets",
    "product_share",
    "recent32_share",
    "recency",
    "position_since_last",
    "seen",
    "last_product",
    "last_target",
    "history_length",
)
CACHE_VERSION = 1


def raw_histories(path):
    owner = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=256, columns=["events"]):
        lists = batch.column(0)
        flat = lists.values
        offsets = lists.offsets.to_numpy()
        events = flat.field("encoded_event").to_numpy()
        products = flat.field("encoded_product").fill_null(-1).to_numpy()
        times = flat.field("timestamp").to_numpy()
        for lo, hi in zip(offsets[:-1], offsets[1:]):
            e, p, t = events[lo:hi], products[lo:hi], times[lo:hi]
            if np.any(np.diff(t) < 0):
                raise ValueError("Unsorted history")
            yield owner, e, p, t
            owner += 1


def dictionary_info(path):
    frame = pd.read_parquet(path)
    count = int(frame.encoded_event.max()) + 1
    kinds = np.zeros(count, dtype=np.int8)
    for row in frame.itertuples(index=False):
        kinds[row.encoded_event] = (
            3 if row.is_target_event else {"view": 1, "click": 2}.get(row.event_type, 0)
        )
    return (
        count,
        kinds,
        set(frame.loc[frame.event_type == "session_boundary", "encoded_event"]),
        set(frame.loc[frame.is_target_event, "encoded_event"]),
    )


def past_positions(events, products, classes, targets):
    eligible = np.isin(events, list(targets))
    cumulative = np.r_[0, np.cumsum(eligible)]
    positions = np.flatnonzero(eligible)
    positions = positions[(positions >= 25) & np.isin(products[positions], classes)]
    return positions[(cumulative[positions] - cumulative[positions - 5]) == 0][-3:]


def encode(events, products, times, cut, classes, kinds, boundaries, max_length):
    e, p, t = events[:cut], products[:cut], times[:cut]
    n = min(max_length, len(e))
    result = {key: np.zeros(max_length, np.int16) for key in TOKENS}
    result["lengths"] = max(1, n)
    feat = np.zeros((len(classes), len(FEATURES)), np.float32)
    feat[:, 6:8] = 1
    result["features"] = feat
    if not len(e):
        result["products"][0] = 1
        return result
    gap = np.r_[0, np.diff(t)]
    result["events"][:n] = e[-n:] + 1
    result["products"][:n] = np.where(p[-n:] < 0, 1, p[-n:] + 3)
    result["gaps"][:n] = (
        1 + np.minimum(25, np.floor(np.log2(1 + gap[-n:] / 1000)))
    ).astype(np.int16)
    result["hours"][:n] = 1 + (t[-n:] // 3_600_000) % 24
    result["weekdays"][:n] = 1 + ((t[-n:] // 86_400_000) + 3) % 7
    result["boundaries"][:n] = 1 + (
        np.isin(e[-n:], list(boundaries)) | (gap[-n:] > 1_800_000)
    )
    match = p[None, :] == classes[:, None]
    count = match.sum(1)
    normalizer = np.log1p(2000)
    feat[:, 0] = np.log1p(count) / normalizer
    for kind in (1, 2, 3):
        feat[:, kind] = (
            np.log1p((match & (kinds[e] == kind)[None, :]).sum(1)) / normalizer
        )
    feat[:, 4] = count / max(1, (p >= 0).sum())
    feat[:, 5] = match[:, -32:].sum(1) / min(32, len(e))
    seen = count > 0
    last = len(e) - 1 - np.argmax(match[:, ::-1], axis=1)
    feat[seen, 6] = np.minimum(
        1, np.log1p((t[-1] - t[last[seen]]) / 3_600_000) / np.log1p(1200)
    )
    feat[seen, 7] = np.log1p(len(e) - 1 - last[seen]) / normalizer
    feat[:, 8] = seen
    positions = np.flatnonzero(p >= 0)
    if len(positions):
        feat[:, 9] = classes == p[positions[-1]]
    positions = np.flatnonzero((kinds[e] == 3) & (p >= 0))
    if len(positions):
        feat[:, 10] = classes == p[positions[-1]]
    feat[:, 11] = np.log1p(len(e)) / normalizer
    return result


def prepare(data_dir, cache_dir, name, classes, max_length=256, log=print):
    data_dir = Path(data_dir)
    path = data_dir / (name + ".parquet")
    destination = Path(cache_dir) / (name + f"_{max_length}")
    destination.mkdir(parents=True, exist_ok=True)
    expected = dict(
        version=CACHE_VERSION,
        source_sha256=sha256(path),
        dictionary_sha256=sha256(data_dir / "event_dictionary.parquet"),
        max_length=max_length,
        classes=np.asarray(classes, dtype=int).tolist(),
    )
    manifest_path = destination / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if any(manifest.get(k) != v for k, v in expected.items()):
            raise ValueError(
                "Cache source/config changed. Remove cache directory explicitly."
            )
        for filename, digest in manifest["files_sha256"].items():
            if sha256(destination / filename) != digest:
                raise ValueError(f"Cache file damaged: {filename}")
        return Store(destination)
    classes = np.asarray(classes, dtype=np.int32)
    if not len(classes) or not np.array_equal(classes, np.unique(classes)):
        raise ValueError("Classes must be nonempty, unique, and sorted")
    event_count, kinds, boundaries, targets = dictionary_info(
        data_dir / "event_dictionary.parquet"
    )
    columns = ["example_id", "client_id"] + (
        ["target_product"] if name == "train" else []
    )
    frame = pd.read_parquet(path, columns=columns)
    if frame.client_id.duplicated().any() or frame.example_id.duplicated().any():
        raise ValueError("Expected unique clients and examples")
    n_original = len(frame)
    last_times = np.zeros(n_original, np.int64)
    additional = []
    product_max = int(classes.max())
    for owner, e, p, t in raw_histories(path):
        last_times[owner] = t[-1] if len(t) else -1
        if len(p):
            product_max = max(product_max, int(p.max()))
        if name == "train":
            positions = past_positions(e, p, classes, targets)
            for j in positions:
                additional.append((owner, int(j) - 5, int(p[j]), 0.5 / len(positions)))
    owners = np.r_[np.arange(n_original), [r[0] for r in additional]].astype(np.int64)
    cuts = np.r_[np.full(n_original, -1), [r[1] for r in additional]].astype(np.int32)
    weights = np.r_[np.ones(n_original), [r[3] for r in additional]].astype(np.float32)
    index = dict(
        ids=frame.example_id.to_numpy(dtype=str),
        clients=frame.client_id.to_numpy(),
        times=last_times,
        owners=owners,
        cuts=cuts,
        weights=weights,
        classes=classes,
    )
    if name == "train":
        labels = np.r_[
            frame.target_product.to_numpy(), [r[2] for r in additional]
        ].astype(np.int32)
        if not np.isin(labels, classes).all():
            raise ValueError("Unexpected target product")
        index["targets"] = labels
    q = len(owners)
    arrays = {
        key: np.lib.format.open_memmap(
            destination / (key + ".npy"),
            mode="w+",
            dtype=np.int16,
            shape=(q, max_length),
        )
        for key in TOKENS
    }
    lengths = np.zeros(q, np.int16)
    features = np.lib.format.open_memmap(
        destination / "features.npy",
        mode="w+",
        dtype=np.float32,
        shape=(q, len(classes), len(FEATURES)),
    )
    extra_by_owner = {}
    for row in range(n_original, q):
        extra_by_owner.setdefault(int(owners[row]), []).append(row)
    for owner, e, p, t in raw_histories(path):
        for row in [owner, *extra_by_owner.get(owner, [])]:
            cut = len(e) if cuts[row] < 0 else int(cuts[row])
            values = encode(e, p, t, cut, classes, kinds, boundaries, max_length)
            for key in TOKENS:
                arrays[key][row] = values[key]
            lengths[row] = values["lengths"]
            features[row] = values["features"]
        if (owner + 1) % 10000 == 0:
            log(f"Prepare {name}: {owner + 1:,}/{n_original:,} clients")
    for array in [*arrays.values(), features]:
        array.flush()
    np.save(destination / "lengths.npy", lengths)
    np.savez_compressed(destination / "index.npz", **index)
    atomic_json(
        manifest_path,
        dict(
            **expected,
            event_count=event_count,
            product_size=product_max + 4,
            original_clients=n_original,
            queries=q,
            files_sha256={p.name: sha256(p) for p in sorted(destination.glob("*.np*"))},
        ),
    )
    return Store(destination)


class Store:
    def __init__(self, path):
        self.path = Path(path)
        self.manifest = json.loads((self.path / "manifest.json").read_text())
        self.tokens = {
            k: np.load(self.path / (k + ".npy"), mmap_mode="r") for k in TOKENS
        }
        self.features = np.load(self.path / "features.npy", mmap_mode="r")
        self.lengths = np.load(self.path / "lengths.npy")
        self.index = dict(np.load(self.path / "index.npz", allow_pickle=False))
        self.ids, self.classes = self.index["ids"], self.index["classes"]
        self.n_original = len(self.ids)
        if "targets" in self.index:
            self.labels = np.searchsorted(self.classes, self.index["targets"]).astype(
                np.int64
            )

    def rows_for(self, owners):
        return np.flatnonzero(np.isin(self.index["owners"], owners))

    def batch(self, rows, context, device):
        if context > self.manifest["max_length"]:
            raise ValueError("Context exceeds cached length")
        lengths = np.minimum(self.lengths[rows], context)
        width = int(max(1, lengths.max()))
        values = {k: np.zeros((len(rows), width), np.int64) for k in TOKENS}
        for i, row in enumerate(rows):
            hi = int(self.lengths[row])
            n = int(lengths[i])
            for key in TOKENS:
                values[key][i, :n] = self.tokens[key][row, hi - n : hi]
        values["lengths"] = lengths.astype(np.int64)
        values["features"] = np.array(self.features[rows])
        return {k: torch.as_tensor(v, device=device) for k, v in values.items()}
