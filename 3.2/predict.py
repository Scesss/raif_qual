"""Reproduce the selected v2 submission from raw test histories and saved models."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool

from gru_click.common import sha256, select_device, atomic_json
from gru_click.data import prepare, dictionary_info
from gru_click.training import load_model, infer
from gru_click.stacking import stack_frame
from src.action_context import load_raw
from src.context_model import context_features
from src.item2vec import load_item_histories
from src.submission import make_submission

ROOT = Path(__file__).resolve().parent


def predict(
    data_dir, models_dir, output, cache_dir, device="auto", threads=4, verify=False
):
    data_dir, models_dir, output, cache_dir = map(
        Path, (data_dir, models_dir, output, cache_dir)
    )
    manifest = json.loads((models_dir / "manifest.json").read_text())
    if manifest.get("experiment") not in ("gru_stack_v2", "calibration_v6"):
        raise ValueError("Unsupported neural artifact bundle")
    if verify and not manifest.get("reference_csv"):
        raise ValueError("This experiment has no frozen reference CSV")
    if (
        manifest.get("reference_csv")
        and output.resolve() == (models_dir / manifest["reference_csv"]).resolve()
    ):
        raise ValueError(
            "Choose an output path different from the frozen reference CSV"
        )
    for filename, digest in manifest["files_sha256"].items():
        if sha256(models_dir / filename) != digest:
            raise ValueError(f"Model checksum mismatch: {filename}")
    if (
        sha256(data_dir / "event_dictionary.parquet")
        != manifest["input_sha256"]["event_dictionary.parquet"]
    ):
        raise ValueError("Event dictionary differs from training")
    device = select_device(device)
    classes = np.asarray(manifest["classes"], np.int32)
    store = prepare(data_dir, cache_dir, "test", classes)
    print("Calculating v7 features from raw test histories", flush=True)
    artifact = joblib.load(models_dir / "v7.joblib")
    np.testing.assert_array_equal(artifact["classes"], classes)
    for filename, digest in artifact["transform_source_sha256"].items():
        if sha256(ROOT / filename) != digest:
            raise ValueError(f"v7 transformation changed: {filename}")
    items = load_item_histories(
        data_dir / "test.parquet",
        data_dir / "event_dictionary.parquet",
        cache_dir / "v7",
        artifact["max_product_events"],
    )
    raw = load_raw(
        data_dir / "test.parquet", cache_dir / "v7", artifact["max_raw_events"]
    )
    np.testing.assert_array_equal(store.ids, items.example_ids)
    frame, categorical = context_features(
        items, raw, np.arange(len(store.ids)), artifact
    )
    v7 = (
        artifact["ranker"]
        .predict(Pool(frame, cat_features=categorical), thread_count=threads)
        .reshape(-1, len(classes))
    )
    del frame, items, raw, artifact
    print(
        f"Running GRU inference on {device}: {store.n_original:,} clients", flush=True
    )
    model = load_model(models_dir / "gru.pt", device)
    if model.config["classes"] != classes.tolist():
        raise ValueError("GRU class order differs from the manifest")
    neural = infer(
        model,
        store,
        np.arange(store.n_original),
        device,
        manifest["context"],
        embeddings=True,
    )
    frame = stack_frame(
        store,
        np.arange(store.n_original),
        neural,
        v7 / manifest["v7_temperature"],
        "scores",
    )
    calibration = manifest.get("calibration")
    if calibration and calibration["head"] == "interest":
        from gru_click.interest import session_statistics

        _, kinds, _, _ = dictionary_info(data_dir / "event_dictionary.parquet")
        extra = session_statistics(store, np.arange(store.n_original), kinds)
        frame = pd.concat(
            [
                frame.reset_index(drop=True),
                extra[[k for k in extra if k.startswith("s_")]].reset_index(drop=True),
            ],
            axis=1,
        )
    ranker = CatBoostRanker()
    ranker.load_model(str(models_dir / "stack.cbm"))
    scores = ranker.predict(
        Pool(frame, cat_features=["candidate"]), thread_count=threads
    ).reshape(-1, len(classes))
    if calibration:
        scores = scores / calibration["temperature"] + np.asarray(calibration["bias"])
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite scores")
    result = make_submission(
        data_dir / "sample_submission.csv",
        store.ids,
        classes[scores.argmax(1)],
        classes,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".csv.tmp")
    result.to_csv(temporary, index=False)
    temporary.replace(output)
    verified = False
    if verify:
        for filename in ("test.parquet", "sample_submission.csv"):
            if sha256(data_dir / filename) != manifest["input_sha256"][filename]:
                raise ValueError(
                    f"Exact replay requires the original input: {filename}"
                )
        reference = (models_dir / manifest["reference_csv"]).resolve()
        if sha256(reference) != manifest["reference_csv_sha256"]:
            raise ValueError("Reference CSV checksum mismatch")
        pd.testing.assert_frame_equal(
            pd.read_csv(output), pd.read_csv(reference), check_dtype=False
        )
        if sha256(output) != manifest["reference_csv_sha256"]:
            raise ValueError("CSV values match but serialization differs")
        verified = True
    report = dict(
        rows=len(result),
        csv_sha256=sha256(output),
        device=device,
        exact_reference_match=verified if verify else None,
        public_accuracy_reference=manifest["public_accuracy"],
        full_retrain_verified=False,
        kaggle_uploads=0,
    )
    atomic_json(output.with_suffix(".json"), report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/submission.csv")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "cache")
    parser.add_argument(
        "--device", choices=["auto", "cpu", "mps", "cuda"], default="auto"
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--verify-reference", action="store_true")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("Positive thread count required")
    predict(
        args.data_dir,
        args.models_dir,
        args.output,
        args.cache_dir,
        args.device,
        args.threads,
        args.verify_reference,
    )


if __name__ == "__main__":
    main()
