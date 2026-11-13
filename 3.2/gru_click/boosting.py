"""GBM trains on clients whose labels were not used in GRU weight updates."""

import time
import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool
from scipy.special import softmax

from .common import atomic_json


def feature_frame(store, rows, neural):
    scores = neural["scores"].astype(np.float32)
    n, c = scores.shape
    users = np.broadcast_to(neural["users"][:, None, :], (n, c, 32))
    context = neural["candidates"]
    probability = softmax(scores, axis=1)
    relative = scores - scores.max(1, keepdims=True)
    rank = np.argsort(np.argsort(-scores, axis=1), axis=1).astype(np.float32) / max(
        1, c - 1
    )
    data = np.concatenate(
        [
            users,
            context,
            np.array(store.features[rows]),
            scores[:, :, None],
            probability[:, :, None],
            relative[:, :, None],
            rank[:, :, None],
        ],
        axis=2,
    ).reshape(n * c, -1)
    frame = pd.DataFrame(data, columns=[f"f{i}" for i in range(data.shape[1])])
    frame.insert(0, "candidate", np.tile(store.classes, n).astype(str))
    return frame


def make_pool(store, rows, neural, labels=True):
    c = len(store.classes)
    y = (
        (np.arange(c)[None, :] == store.labels[rows, None]).astype(np.int8).ravel()
        if labels
        else None
    )
    return Pool(
        feature_frame(store, rows, neural),
        label=y,
        cat_features=["candidate"],
        group_id=np.repeat(np.arange(len(rows)), c),
    )


class TimeLimit:
    def __init__(self, deadline, update):
        self.deadline, self.update = deadline, update

    def after_iteration(self, info):
        if info.iteration % 25 == 0:
            self.update(gbm_iteration=info.iteration)
        return time.time() < self.deadline


def fit_gbm(
    store,
    rows,
    neural,
    validation,
    validation_neural,
    destination,
    *,
    iterations=600,
    deadline,
    threads=4,
    update=lambda **kw: None,
):
    if time.time() >= deadline:
        raise TimeoutError(
            "No remaining GBM training budget; existing backup files remain available"
        )
    model = CatBoostRanker(
        loss_function="YetiRank",
        eval_metric="NDCG:top=1",
        iterations=iterations,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=8,
        random_seed=42,
        thread_count=threads,
        task_type="CPU",
        allow_writing_files=False,
    )
    train_pool = make_pool(store, rows, neural)
    kwargs = dict(verbose=100, callbacks=[TimeLimit(deadline, update)])
    if validation is not None:
        kwargs.update(
            eval_set=make_pool(store, validation, validation_neural),
            early_stopping_rounds=70,
            use_best_model=True,
        )
    model.fit(train_pool, **kwargs)
    temporary = destination.with_suffix(".cbm.tmp")
    model.save_model(str(temporary))
    temporary.replace(destination)
    atomic_json(
        destination.with_suffix(".json"),
        dict(
            trees=model.tree_count_,
            fit_clients=len(rows),
            features=model.feature_names_,
            loss="YetiRank",
            early_stopping=validation is not None,
        ),
    )
    return model


def gbm_scores(model, store, rows, neural, threads=4):
    scores = model.predict(
        make_pool(store, rows, neural, labels=False), thread_count=threads
    ).reshape(len(rows), len(store.classes))
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite CatBoost scores")
    return scores
