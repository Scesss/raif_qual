"""Causal stacking with earlier v7 predictions and frozen GRU representations."""

import numpy as np
from catboost import CatBoostRanker, Pool
from scipy.special import softmax

from .boosting import feature_frame, TimeLimit
from .common import atomic_json


def align_scores(indices, scores, requested):
    if len(np.unique(indices)) != len(indices):
        raise ValueError("Duplicate score owners")
    order = np.argsort(indices)
    positions = np.searchsorted(indices[order], requested)
    if (positions >= len(indices)).any() or not np.array_equal(
        indices[order][positions], requested
    ):
        raise ValueError(
            "Requested clients are missing from saved out-of-sample scores"
        )
    return scores[order[positions]]


def stack_frame(store, rows, neural, baseline, mode="embedding"):
    frame = feature_frame(store, rows, neural)
    if mode == "scores":
        frame = frame[["candidate", *[f"f{i}" for i in range(64, 80)]]].copy()
    elif mode != "embedding":
        raise ValueError("Unknown stacking mode")
    probability = softmax(baseline, axis=1)
    rank = np.argsort(np.argsort(-baseline, axis=1), axis=1).astype(np.float32) / max(
        1, len(store.classes) - 1
    )
    frame["v7_probability"] = probability.ravel().astype(np.float32)
    frame["v7_rank"] = rank.ravel()
    frame["v7_relative_score"] = (
        (baseline - baseline.max(1, keepdims=True)).ravel().astype(np.float32)
    )
    frame["v7_top"] = (rank == 0).ravel().astype(np.float32)
    # Global uncertainty lets a tree condition reliance on GRU on v7 confidence.
    sorted_p = np.sort(probability, axis=1)
    margin = sorted_p[:, -1] - sorted_p[:, -2]
    frame["v7_margin"] = np.repeat(margin, len(store.classes)).astype(np.float32)
    return frame


def stack_pool(store, rows, neural, baseline, mode, labels=True):
    c = len(store.classes)
    y = (
        (np.arange(c)[None, :] == store.labels[rows, None]).astype(np.int8).ravel()
        if labels
        else None
    )
    return Pool(
        stack_frame(store, rows, neural, baseline, mode),
        label=y,
        cat_features=["candidate"],
        group_id=np.repeat(np.arange(len(rows)), c),
    )


def train_stack(
    store,
    rows,
    neural,
    baseline,
    validation,
    validation_neural,
    validation_baseline,
    path,
    *,
    mode="embedding",
    depth=4,
    iterations=500,
    deadline,
    update=lambda **kw: None,
):
    model = CatBoostRanker(
        loss_function="YetiRank",
        eval_metric="NDCG:top=1",
        iterations=iterations,
        depth=depth,
        learning_rate=0.04,
        l2_leaf_reg=10,
        random_seed=42,
        thread_count=4,
        task_type="CPU",
        allow_writing_files=False,
    )
    kwargs = dict(verbose=100, callbacks=[TimeLimit(deadline, update)])
    if validation is not None:
        kwargs.update(
            eval_set=stack_pool(
                store, validation, validation_neural, validation_baseline, mode
            ),
            early_stopping_rounds=60,
            use_best_model=True,
        )
    model.fit(stack_pool(store, rows, neural, baseline, mode), **kwargs)
    temporary = path.with_suffix(".cbm.tmp")
    model.save_model(str(temporary))
    temporary.replace(path)
    atomic_json(
        path.with_suffix(".json"),
        dict(
            mode=mode,
            depth=depth,
            trees=model.tree_count_,
            fit_clients=len(rows),
            validation_clients=0 if validation is None else len(validation),
            v7_features_are_oof=True,
        ),
    )
    return model


def predict_stack(model, store, rows, neural, baseline, mode):
    scores = model.predict(
        stack_pool(store, rows, neural, baseline, mode, labels=False), thread_count=4
    ).reshape(len(rows), len(store.classes))
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite stacked predictions")
    return scores
