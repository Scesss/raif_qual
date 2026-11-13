"""The retained TF-IDF baseline and session Item2Vec ranker."""

from types import SimpleNamespace
import joblib
import numpy as np
import pandas as pd
from catboost import Pool
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from gru_click.common import atomic_json
from src.features import load_histories, model_inputs
from src.model import build_model
from src.item2vec import (
    load_item_histories,
    train_embedding,
    candidate_features,
    candidate_labels,
)
from src.validation import chronological_split
from src.submission import make_submission
from .context import ranker


def train_tfidf(data_dir, out, config, threads):
    dictionary = pd.read_parquet(data_dir / "event_dictionary.parquet")
    target_codes = set(dictionary.loc[dictionary.is_target_event, "encoded_event"])
    train = load_histories(
        data_dir / "train.parquet", out / "cache", target_codes, config["max_events"]
    )
    test = load_histories(
        data_dir / "test.parquet", out / "cache", target_codes, config["max_events"]
    )
    args = SimpleNamespace(**config)
    rows = np.arange(len(train.targets))
    fit, check = train_test_split(
        rows,
        test_size=config["validation_size"],
        random_state=42,
        stratify=train.targets,
    )
    with threadpool_limits(limits=threads):
        model = build_model(args)
        model.fit(model_inputs(train, "baseline", fit), train.targets[fit])
        accuracy = float(
            np.mean(
                model.predict(model_inputs(train, "baseline", check))
                == train.targets[check]
            )
        )
        model = build_model(args)
        model.fit(model_inputs(train, "baseline"), train.targets)
        prediction = model.predict(model_inputs(test, "baseline"))
    joblib.dump(model, out / "tfidf.joblib", compress=3)
    classes = np.unique(train.targets)
    make_submission(
        data_dir / "sample_submission.csv", test.example_ids, prediction, classes
    ).to_csv(out / "submission.csv", index=False)
    atomic_json(
        out / "metrics.json",
        dict(
            accuracy=accuracy,
            split="stratified_80_20_seed42",
            full_retrain_verified=False,
            kaggle_uploads=0,
        ),
    )


def train_item2vec(data_dir, out, config, threads):
    train = load_item_histories(
        data_dir / "train.parquet",
        data_dir / "event_dictionary.parquet",
        out / "cache",
        2000,
    )
    test = load_item_histories(
        data_dir / "test.parquet",
        data_dir / "event_dictionary.parquet",
        out / "cache",
        2000,
    )
    classes = np.unique(train.targets)
    fit, check, cutoff = chronological_split(
        train.last_timestamps, train.client_ids, 0.2
    )
    embedding, _ = train_embedding(
        train, fit, classes, config["dimensions"], config["epochs"], 42
    )
    a, _, _ = candidate_features(train, fit, classes, embedding)
    b, _, _ = candidate_features(train, check, classes, embedding)

    def pool(frame, y):
        labels, groups = candidate_labels(y, classes)
        return Pool(
            frame, label=labels, group_id=groups, cat_features=["candidate_product"]
        )

    model = ranker(config["iterations"], threads, 64)
    model.fit(
        pool(a, train.targets[fit]),
        eval_set=pool(b, train.targets[check]),
        early_stopping_rounds=60,
        use_best_model=True,
    )
    values = model.predict(Pool(b, cat_features=["candidate_product"])).reshape(
        -1, len(classes)
    )
    accuracy = float(np.mean(classes[values.argmax(1)] == train.targets[check]))
    trees = model.tree_count_
    embedding, _ = train_embedding(
        train,
        np.arange(len(train.targets)),
        classes,
        config["dimensions"],
        config["epochs"],
        42,
    )
    a, _, _ = candidate_features(
        train, np.arange(len(train.targets)), classes, embedding
    )
    model = ranker(trees, threads, 64)
    model.fit(pool(a, train.targets))
    b, _, _ = candidate_features(
        test, np.arange(len(test.histories)), classes, embedding
    )
    prediction = classes[
        model.predict(Pool(b, cat_features=["candidate_product"]))
        .reshape(-1, len(classes))
        .argmax(1)
    ]
    model.save_model(str(out / "ranker.cbm"))
    joblib.dump(embedding, out / "embedding.joblib")
    make_submission(
        data_dir / "sample_submission.csv", test.example_ids, prediction, classes
    ).to_csv(out / "submission.csv", index=False)
    atomic_json(
        out / "metrics.json",
        dict(
            accuracy=accuracy,
            split="chronological_80_20",
            cutoff_timestamp_ms=cutoff,
            trees=trees,
            full_retrain_verified=False,
            kaggle_uploads=0,
        ),
    )
