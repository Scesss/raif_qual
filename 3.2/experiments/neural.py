"""Fixed GRU/v7 stacking recipe; no automatic architecture or wall-clock cutoff."""

import gc
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostRanker, Pool

from gru_click.common import atomic_json, sha256, select_device
from gru_click.data import prepare, dictionary_info
from gru_click.training import fit_gru, infer
from gru_click.stacking import stack_frame, train_stack, predict_stack
from gru_click.evaluation import compare, temperature
from gru_click.interest import session_statistics
from gru_click.calibration import fit_bias
from .context import train_context, earlier_scores
from .protocol import split_roles

ROOT = Path(__file__).resolve().parents[1]


def candidate_pool(frame, labels, c):
    y = (
        None
        if labels is None
        else (np.arange(c)[None, :] == labels[:, None]).astype(np.int8).ravel()
    )
    return Pool(
        frame,
        label=y,
        cat_features=["candidate"],
        group_id=np.repeat(np.arange(len(frame) // c), c),
    )


def interest_frame(store, rows, neural, baseline, kinds):
    frame = stack_frame(store, rows, neural, baseline, "scores")
    extra = session_statistics(store, rows, kinds)
    return pd.concat(
        [
            frame.reset_index(drop=True),
            extra[[k for k in extra if k.startswith("s_")]].reset_index(drop=True),
        ],
        axis=1,
    )


def train_neural(data_dir, out, config, threads, requested_device, calibration=False):
    device = select_device(requested_device)
    original = json.loads((ROOT / "models/manifest.json").read_text())
    classes = np.asarray(original["classes"], np.int32)
    train_context(data_dir, out / "context", threads)
    store = prepare(data_dir, out / "cache", "train", classes)
    split = split_roles(store)
    np.savez_compressed(out / "split.npz", **split)
    baseline, v7_t, audit = earlier_scores(out / "context", store, split)
    atomic_json(out / "oof_audit.json", dict(provenance=audit, labels_excluded=True))
    network = dict(
        event_count=store.manifest["event_count"],
        product_size=store.manifest["product_size"],
        classes=classes.tolist(),
        width=config["width"],
        layers=config["layers"],
        dropout=config["dropout"],
    )
    local_path = out / "local_gru.pt"
    model, local_state = fit_gru(
        store,
        store.rows_for(split["gru_fit"]),
        split["early_stop"],
        network,
        local_path,
        device=device,
        context=config["context"],
        batch_size=config["batch_size"],
        epochs=config["local_epochs"],
        learning_rate=config["local_lr"],
        deadline=float("inf"),
    )
    encoded = {
        r: infer(model, store, split[r], device, config["context"], embeddings=True)
        for r in ["gbm_fit", "early_stop", "calibration", "check"]
    }
    del model
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    local = train_stack(
        store,
        split["gbm_fit"],
        encoded["gbm_fit"],
        baseline["gbm_fit"],
        split["early_stop"],
        encoded["early_stop"],
        baseline["early_stop"],
        out / "local_stack.cbm",
        mode="scores",
        depth=4,
        iterations=500,
        deadline=float("inf"),
    )
    reference_check = predict_stack(
        local, store, split["check"], encoded["check"], baseline["check"], "scores"
    )
    selected_bias = None
    if calibration:
        selected_bias = local_calibration(
            store, split, encoded, baseline, local, data_dir, out
        )
    atomic_json(
        out / "local_metrics.json",
        dict(
            accuracy=float(
                np.mean(reference_check.argmax(1) == store.labels[split["check"]])
            ),
            split="last_temporal_10k_reused",
            training=local_state,
            selected_original_final_trees=config["iterations"],
        ),
    )
    # Final network is a continuation from this run's selected local checkpoint, never the shipped final weights.
    full_path = out / "full_gru.pt"
    model, full_state = fit_gru(
        store,
        store.rows_for(split["full_gru"]),
        None,
        network,
        full_path,
        device=device,
        context=config["context"],
        batch_size=config["batch_size"],
        epochs=config["full_epochs"],
        learning_rate=config["full_lr"],
        deadline=float("inf"),
        initial=local_path,
    )
    full_encoded = infer(
        model, store, split["full_gbm"], device, config["context"], embeddings=True
    )
    del model, encoded, local
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    models = out / "models"
    models.mkdir()
    shutil.copy2(full_path, models / "gru.pt")
    shutil.copy2(out / "context/v7.joblib", models / "v7.joblib")
    stack_path = models / "stack.cbm"
    manifest = dict(
        original,
        files_sha256={},
        v7_temperature=v7_t,
        public_accuracy=None,
        public_accuracy_source=None,
        original_public_accuracy=0.41172,
        full_retrain_verified=False,
        original_training=dict(
            backend=device,
            seed=42,
            local_epochs=config["local_epochs"],
            full_epochs=config["full_epochs"],
        ),
        reference_csv=os.path.relpath(ROOT / "results/submission_v2.csv", models),
    )
    if not calibration or selected_bias["head"] == "reference":
        train_stack(
            store,
            split["full_gbm"],
            full_encoded,
            baseline["full_gbm"],
            None,
            None,
            None,
            stack_path,
            mode="scores",
            depth=4,
            iterations=config["iterations"],
            deadline=float("inf"),
        )
    else:
        _, kinds, _, _ = dictionary_info(data_dir / "event_dictionary.parquet")
        frame = interest_frame(
            store, split["full_gbm"], full_encoded, baseline["full_gbm"], kinds
        )
        ranker = CatBoostRanker(
            loss_function="YetiRank",
            eval_metric="NDCG:top=1",
            depth=3,
            iterations=selected_bias["interest_trees"],
            learning_rate=0.04,
            l2_leaf_reg=10,
            random_seed=42,
            task_type="CPU",
            thread_count=2,
            allow_writing_files=False,
        )
        ranker.fit(
            candidate_pool(frame, store.labels[split["full_gbm"]], len(classes)),
            verbose=100,
        )
        ranker.save_model(str(stack_path))
    if calibration:
        manifest.update(
            experiment="calibration_v6",
            calibration=selected_bias,
            reference_csv=None,
            reference_csv_sha256=None,
        )
    manifest["files_sha256"] = {
        name: sha256(models / name) for name in ["gru.pt", "v7.joblib", "stack.cbm"]
    }
    atomic_json(models / "manifest.json", manifest)
    from predict import predict

    predict(data_dir, models, out / "submission.csv", out / "cache", device, threads)
    predicted = pd.read_csv(out / "submission.csv")
    expected = pd.read_csv(ROOT / "results/submission_v2.csv")
    match = predicted.equals(expected)
    atomic_json(
        out / "training_report.json",
        dict(
            experiment=manifest["experiment"],
            local_training=local_state,
            full_training=full_state,
            actual_csv_matches_selected_v2=match,
            changed_from_selected_v2=int(
                np.sum(
                    predicted.encoded_product.to_numpy()
                    != expected.encoded_product.to_numpy()
                )
            ),
            csv_sha256=sha256(out / "submission.csv"),
            full_retrain_verified=match and not calibration,
            note="This is a new training run, not a claim about the unexecuted submission-package preparation.",
            kaggle_uploads=0,
        ),
    )


def local_calibration(store, split, encoded, baseline, control, data_dir, out):
    _, kinds, _, _ = dictionary_info(data_dir / "event_dictionary.parquet")
    frames = {
        r: interest_frame(store, split[r], encoded[r], baseline[r], kinds)
        for r in encoded
    }
    c = len(store.classes)
    model = CatBoostRanker(
        loss_function="YetiRank",
        eval_metric="NDCG:top=1",
        depth=3,
        iterations=500,
        learning_rate=0.04,
        l2_leaf_reg=10,
        random_seed=42,
        thread_count=2,
        task_type="CPU",
        allow_writing_files=False,
    )
    model.fit(
        candidate_pool(frames["gbm_fit"], store.labels[split["gbm_fit"]], c),
        eval_set=candidate_pool(
            frames["early_stop"], store.labels[split["early_stop"]], c
        ),
        early_stopping_rounds=60,
        use_best_model=True,
        verbose=100,
    )
    model.save_model(str(out / "local_interest.cbm"))

    def predict_one(which, role):
        return which.predict(
            candidate_pool(frames[role][which.feature_names_], None, c), thread_count=2
        ).reshape(-1, c)

    candidates = []
    recipes = {}
    reference_cal = predict_one(control, "calibration").argmax(1)
    for head, ranker in [("reference", control), ("interest", model)]:
        early = predict_one(ranker, "early_stop")
        cal = predict_one(ranker, "calibration")
        t = temperature(early, store.labels[split["early_stop"]])
        variants = [(head + "_unchanged", np.zeros(c))]
        for penalty in [0.1, 1.0, 10.0]:
            bias = fit_bias(early / t, store.labels[split["early_stop"]], penalty)
            variants.extend((f"{head}_l2_{penalty}_w{w}", bias * w) for w in [0.5, 1.0])
        for name, bias in variants:
            pred = (cal / t + bias).argmax(1)
            candidates.append(
                dict(
                    name=name,
                    accuracy=float(np.mean(pred == store.labels[split["calibration"]])),
                    changes=int(np.sum(pred != reference_cal)),
                )
            )
            recipes[name] = dict(
                head=head,
                temperature=t,
                bias=bias.tolist(),
                interest_trees=model.tree_count_,
            )
    selected = max(candidates, key=lambda v: (v["accuracy"], -v["changes"]))
    recipe = recipes[selected["name"]]
    atomic_json(
        out / "frozen_calibration.json",
        dict(
            candidates=candidates,
            selected=selected,
            recipe=recipe,
            check_not_used_for_selection=True,
        ),
    )
    check = predict_one(
        model if recipe["head"] == "interest" else control, "check"
    ) / recipe["temperature"] + np.array(recipe["bias"])
    reference_check = predict_one(control, "check")
    atomic_json(
        out / "calibration_metrics.json",
        compare(
            check,
            reference_check,
            store.labels[split["check"]],
            store.features[split["check"], :, 8],
        ),
    )
    return recipe
