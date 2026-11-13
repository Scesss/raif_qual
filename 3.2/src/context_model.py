"""Shared train/inference feature schema for the selected action-context ranker."""

import numpy as np
import pandas as pd
from catboost import Pool

from src.action_context import action_features, relation_features
from src.item2vec import candidate_features

CONTEXT_MODEL_VERSION = 1


def dictionary_context(path):
    dictionary = pd.read_parquet(path)
    event_count = int(dictionary.encoded_event.max()) + 1
    kinds = np.zeros(event_count, dtype=np.int8)
    for row in dictionary.itertuples(index=False):
        kinds[row.encoded_event] = 3 if row.is_target_event else {"view": 1, "click": 2}.get(row.event_type, 0)
    boundaries = set(dictionary.loc[dictionary.event_type == "session_boundary", "encoded_event"])
    targets = set(dictionary.loc[dictionary.is_target_event, "encoded_event"])
    return event_count, kinds, boundaries, targets


def context_features(items, raw, indices, artifact, log=print):
    if not np.array_equal(items.example_ids, raw.example_ids):
        raise ValueError("Raw and product histories are not aligned")
    classes = artifact["classes"]
    base, users, cosine = candidate_features(items, indices, classes, artifact["embedding"], log)
    del users, cosine
    actions, action_cats = action_features(raw, indices, classes, artifact["schema"], artifact["kinds"],
                                          artifact["boundaries"], log)
    relations = relation_features(items, raw, indices, classes, artifact["tables"])
    frame = pd.concat([base, actions, relations], axis=1, copy=False)
    categorical = ["candidate_product"] + action_cats
    if frame.columns.duplicated().any():
        raise ValueError("Duplicate candidate feature names")
    if "feature_names" in artifact and frame.columns.tolist() != artifact["feature_names"]:
        raise ValueError("Feature schema differs from the saved model")
    if "categorical" in artifact and categorical != artifact["categorical"]:
        raise ValueError("Categorical schema differs from the saved model")
    return frame, categorical


def context_predictions(items, raw, artifact, threads=4, log=print):
    if artifact["model_type"] != "action_context_ranker" or artifact["feature_version"] != CONTEXT_MODEL_VERSION:
        raise ValueError("Unsupported action-context model")
    frame, categorical = context_features(items, raw, np.arange(len(items.histories)), artifact, log)
    scores = artifact["ranker"].predict(Pool(frame, cat_features=categorical), thread_count=threads)
    return artifact["classes"][scores.reshape(len(items.histories), -1).argmax(axis=1)]
