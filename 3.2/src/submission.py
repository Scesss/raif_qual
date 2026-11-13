"""Submission alignment and integrity checks, independent of the model."""

from pathlib import Path

import numpy as np
import pandas as pd


def make_submission(
    sample_path: Path,
    example_ids: np.ndarray,
    predictions: np.ndarray,
    allowed_products: np.ndarray,
) -> pd.DataFrame:
    sample = pd.read_csv(sample_path, dtype={"example_id": str})
    if list(sample.columns) != ["example_id", "encoded_product"]:
        raise ValueError("Unexpected sample_submission columns")
    if len(example_ids) != len(predictions):
        raise ValueError("Different numbers of IDs and predictions")
    if sample.example_id.duplicated().any() or len(set(example_ids)) != len(example_ids):
        raise ValueError("Duplicate example_id")
    if set(sample.example_id) != set(example_ids):
        raise ValueError("Test IDs do not match sample_submission")
    if not np.isin(predictions, allowed_products).all():
        raise ValueError("Predicted product is outside the training classes")
    # The sample and test have DIFFERENT row orders. Always align by example_id.
    indexed = pd.Series(predictions, index=example_ids)
    sample["encoded_product"] = indexed.loc[sample.example_id].to_numpy(dtype=np.int32)
    if sample.isna().any().any():
        raise ValueError("Missing submission values")
    return sample
