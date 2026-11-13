"""Rebuild the original v7 ranker and its strictly earlier fold predictions."""

import gc
from pathlib import Path
import joblib
import numpy as np
from catboost import CatBoostRanker, Pool

from gru_click.common import atomic_json, sha256
from gru_click.evaluation import temperature
from gru_click.stacking import align_scores
from src.action_context import load_raw, fit_context_schema, fit_relations
from src.context_model import (
    dictionary_context,
    context_features,
    CONTEXT_MODEL_VERSION,
)
from src.item2vec import load_item_histories, train_embedding, candidate_labels
from .protocol import rolling_blocks

ROOT = Path(__file__).resolve().parents[1]


def ranker(iterations, threads, one_hot=2048):
    return CatBoostRanker(
        loss_function="YetiRank:mode=NDCG;top=1",
        eval_metric="NDCG:top=1",
        iterations=iterations,
        learning_rate=0.07,
        depth=6,
        l2_leaf_reg=8,
        border_count=64,
        random_seed=42,
        thread_count=threads,
        task_type="CPU",
        one_hot_max_size=one_hot,
        allow_writing_files=False,
        verbose=100,
    )


def load_training(data_dir, cache):
    items = load_item_histories(
        data_dir / "train.parquet", data_dir / "event_dictionary.parquet", cache, 2000
    )
    raw = load_raw(data_dir / "train.parquet", cache, 256)
    np.testing.assert_array_equal(items.example_ids, raw.example_ids)
    return items, raw


def fit_artifact(items, raw, fit, early, data_dir, path, threads, iterations):
    classes = np.unique(items.targets)
    embedding, w2v = train_embedding(items, fit, classes, 24, 8, 42)
    del w2v
    count, kinds, boundaries, targets = dictionary_context(
        data_dir / "event_dictionary.parquet"
    )
    artifact = dict(
        model_type="action_context_ranker",
        feature_version=CONTEXT_MODEL_VERSION,
        classes=classes,
        embedding=embedding,
        schema=fit_context_schema(raw, fit, count),
        tables=fit_relations(items, raw, fit, classes, count, targets),
        kinds=kinds,
        boundaries=boundaries,
        max_product_events=2000,
        max_raw_events=256,
        dictionary_sha256=sha256(data_dir / "event_dictionary.parquet"),
        transform_source_sha256={
            name: sha256(ROOT / name)
            for name in [
                "src/action_context.py",
                "src/item2vec.py",
                "src/context_model.py",
            ]
        },
    )
    frame, categorical = context_features(items, raw, fit, artifact)
    artifact.update(feature_names=frame.columns.tolist(), categorical=categorical)
    labels, groups = candidate_labels(items.targets[fit], classes)
    train_pool = Pool(frame, label=labels, group_id=groups, cat_features=categorical)
    del frame
    model = ranker(iterations, threads)
    if early is not None:
        frame, _ = context_features(items, raw, early, artifact)
        labels, groups = candidate_labels(items.targets[early], classes)
        model.fit(
            train_pool,
            eval_set=Pool(
                frame, label=labels, group_id=groups, cat_features=categorical
            ),
            early_stopping_rounds=60,
            use_best_model=True,
        )
        del frame
    else:
        model.fit(train_pool)
    artifact["ranker"] = model
    joblib.dump(artifact, path, compress=3)
    return artifact


def scores(artifact, items, raw, rows, threads):
    frame, categorical = context_features(items, raw, rows, artifact)
    return (
        artifact["ranker"]
        .predict(Pool(frame, cat_features=categorical), thread_count=threads)
        .reshape(-1, len(artifact["classes"]))
    )


def train_context(data_dir, out, threads, items=None, raw=None):
    out.mkdir(parents=True, exist_ok=True)
    if items is None:
        items, raw = load_training(data_dir, out / "cache")
    for fold, fraction in enumerate([0.6, 0.7, 0.8], 1):
        directory = out / f"fold_{fold}"
        directory.mkdir(exist_ok=True)
        blocks = rolling_blocks(items.last_timestamps, items.client_ids, fraction)
        np.savez_compressed(directory / "split.npz", **blocks)
        if (directory / "scores.npz").exists():
            continue
        print(f"v7 fold {fold}: {len(blocks['fit']):,} training clients", flush=True)
        artifact = fit_artifact(
            items,
            raw,
            blocks["fit"],
            blocks["early_stop"],
            data_dir,
            directory / "v7.joblib",
            threads,
            350,
        )
        calibration = scores(artifact, items, raw, blocks["calibration"], threads)
        check = scores(artifact, items, raw, blocks["check"], threads)
        t = temperature(
            calibration,
            np.searchsorted(artifact["classes"], items.targets[blocks["calibration"]]),
        )
        np.savez_compressed(
            directory / "scores.npz",
            classes=artifact["classes"],
            check_indices=blocks["check"],
            calibration_scores=calibration,
            check_scores=check,
        )
        atomic_json(
            directory / "metrics.json",
            dict(
                temperature=t,
                trees=artifact["ranker"].tree_count_,
                accuracy=float(
                    np.mean(
                        artifact["classes"][check.argmax(1)]
                        == items.targets[blocks["check"]]
                    )
                ),
                fit_clients=len(blocks["fit"]),
                check_clients=len(blocks["check"]),
            ),
        )
        del artifact
        gc.collect()
    full = out / "v7.joblib"
    if not full.exists():
        print("v7 full model: all original training clients, 338 trees", flush=True)
        fit_artifact(
            items,
            raw,
            np.arange(len(items.histories)),
            None,
            data_dir,
            full,
            threads,
            338,
        )
    return full


def earlier_scores(context_dir, store, split):
    result = {}
    audit = []
    for fold, roles in [
        (1, ["gbm_fit"]),
        (2, ["early_stop", "calibration"]),
        (3, ["check", "full_gbm"]),
    ]:
        folder = context_dir / f"fold_{fold}"
        blocks = dict(np.load(folder / "split.npz"))
        saved = dict(np.load(folder / "scores.npz"))
        np.testing.assert_array_equal(saved["classes"], store.classes)
        t = temperature(
            saved["calibration_scores"], store.labels[blocks["calibration"]]
        )
        used = np.r_[blocks["fit"], blocks["early_stop"], blocks["calibration"]]
        for role in roles:
            rows = split[role]
            if (
                np.isin(rows, used).any()
                or store.index["times"][used].max() >= store.index["times"][rows].min()
            ):
                raise ValueError(f"OOF temporal overlap for {role}")
            result[role] = (
                align_scores(saved["check_indices"], saved["check_scores"], rows) / t
            )
            audit.append(
                dict(
                    role=role,
                    fold=fold,
                    labels_excluded=True,
                    prediction_clients=len(rows),
                )
            )
    return result, t, audit
