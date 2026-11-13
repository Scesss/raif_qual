"""Train one retained experiment from supplied Kaggle data. Never uploads predictions."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")

from gru_click.common import atomic_json, sha256

ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ("tfidf", "item2vec", "catboost_v7", "gru_stack_v2", "calibration_v6")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=EXPERIMENTS, default="gru_stack_v2")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/trained_v2")
    parser.add_argument(
        "--device", choices=["auto", "mps", "cuda", "cpu"], default="auto"
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the fixed recipe without preparing data or training",
    )
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("Positive thread count required")
    config = json.loads((ROOT / "configs" / f"{args.experiment}.json").read_text())
    if args.dry_run:
        print(
            json.dumps(
                dict(
                    experiment=args.experiment,
                    config=config,
                    training_started=False,
                    kaggle_uploads=0,
                ),
                indent=2,
            )
        )
        return
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(
            "Output is not empty; choose a new directory to avoid mixing training runs"
        )
    manifest = json.loads((ROOT / "models/manifest.json").read_text())
    for name, digest in manifest["input_sha256"].items():
        if not (args.data_dir / name).exists():
            parser.error(f"Missing input: {name}")
        if sha256(args.data_dir / name) != digest:
            parser.error(f"Original experiment input required: {name}")
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(
        args.output / "protocol.json",
        dict(
            experiment=args.experiment,
            config=config,
            device=args.device,
            input_sha256=manifest["input_sha256"],
            full_retrain_verified=False,
            kaggle_uploads=0,
        ),
    )
    if args.experiment in ("tfidf", "item2vec"):
        from experiments.linear import train_tfidf, train_item2vec

        (train_tfidf if args.experiment == "tfidf" else train_item2vec)(
            args.data_dir, args.output, config, args.threads
        )
    elif args.experiment == "catboost_v7":
        from experiments.context import train_context, scores
        import joblib, numpy as np
        from src.item2vec import load_item_histories
        from src.action_context import load_raw
        from src.submission import make_submission

        path = train_context(args.data_dir, args.output, args.threads)
        artifact = joblib.load(path)
        items = load_item_histories(
            args.data_dir / "test.parquet",
            args.data_dir / "event_dictionary.parquet",
            args.output / "cache",
            2000,
        )
        raw = load_raw(args.data_dir / "test.parquet", args.output / "cache", 256)
        values = scores(
            artifact, items, raw, np.arange(len(items.histories)), args.threads
        )
        make_submission(
            args.data_dir / "sample_submission.csv",
            items.example_ids,
            artifact["classes"][values.argmax(1)],
            artifact["classes"],
        ).to_csv(args.output / "submission.csv", index=False)
    else:
        from experiments.neural import train_neural

        base = json.loads((ROOT / "configs/gru_stack_v2.json").read_text())
        train_neural(
            args.data_dir,
            args.output,
            base,
            args.threads,
            args.device,
            args.experiment == "calibration_v6",
        )


if __name__ == "__main__":
    main()
