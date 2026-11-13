"""Small, offline checks. No full training or external API requests."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from gru_click.common import sha256
from gru_click.data import encode, past_positions, prepare
from gru_click.model import GRUProductModel
from gru_click.stacking import align_scores, stack_frame
from gru_click.calibration import fit_bias
from experiments.protocol import rolling_blocks, split_roles
from src.submission import make_submission

ROOT = Path(__file__).resolve().parents[1]


class SubmissionTests(unittest.TestCase):
    def test_model_bundle_checksums(self):
        manifest = json.loads((ROOT / "models/manifest.json").read_text())
        for name, digest in manifest["files_sha256"].items():
            self.assertEqual(sha256(ROOT / "models" / name), digest)
        self.assertEqual(
            sha256(ROOT / "results/submission_v2.csv"), manifest["reference_csv_sha256"]
        )

    def test_submission_aligns_sample_instead_of_test_order(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "sample.csv"
            pd.DataFrame({"example_id": ["b", "a"], "encoded_product": [0, 0]}).to_csv(
                path, index=False
            )
            frame = make_submission(
                path, np.array(["a", "b"]), np.array([3, 7]), np.array([3, 7])
            )
            self.assertEqual(frame.encoded_product.tolist(), [7, 3])
            with self.assertRaises(ValueError):
                make_submission(
                    path, np.array(["a", "a"]), np.array([3, 7]), np.array([3, 7])
                )

    def test_oof_alignment_and_missing_clients(self):
        z = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        np.testing.assert_array_equal(
            align_scores(np.array([20, 10, 30]), z, np.array([30, 10])), z[[2, 1]]
        )
        with self.assertRaises(ValueError):
            align_scores(np.array([20, 10, 30]), z, np.array([40]))

    def test_temporal_roles_are_disjoint_and_ties_stay_together(self):
        from types import SimpleNamespace

        times = np.arange(1000, dtype=np.int64) * 100
        store = SimpleNamespace(
            index=dict(times=times, clients=np.arange(1000)), n_original=1000
        )
        roles = split_roles(store)
        self.assertEqual(
            [
                len(roles[k])
                for k in ["gru_fit", "gbm_fit", "early_stop", "calibration", "check"]
            ],
            [700, 100, 50, 50, 100],
        )
        self.assertEqual(len(np.intersect1d(roles["full_gru"], roles["full_gbm"])), 0)
        tied = np.repeat(np.arange(100), 10)
        blocks = rolling_blocks(tied, np.arange(1000), 0.8)
        self.assertLess(tied[blocks["fit"]].max(), tied[blocks["early_stop"]].min())

    def test_hidden_actions_do_not_change_prefix_inputs(self):
        events = np.arange(40) % 5
        products = np.arange(40) % 2 + 3
        times = np.arange(40) * 1000
        kinds = np.array([0, 1, 2, 3, 0])
        classes = np.array([3, 4])
        a = encode(events, products, times, 25, classes, kinds, {4}, 32)
        events[25:] = 4
        products[25:] = 99
        times[25:] += 100000
        b = encode(events, products, times, 25, classes, kinds, {4}, 32)
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])

    def test_historical_targets_exclude_the_hidden_five_events(self):
        events = np.zeros(60, dtype=int)
        events[[30, 34, 50]] = 3
        products = np.full(60, 3)
        self.assertEqual(
            past_positions(events, products, np.array([3]), {3}).tolist(), [30, 50]
        )

    def test_empty_history_and_forward_padding(self):
        classes = np.array([3, 7])
        kinds = np.array([0, 1, 3, 0, 2, 0])
        empty = encode(
            np.array([], int),
            np.array([], int),
            np.array([], int),
            0,
            classes,
            kinds,
            {3},
            16,
        )
        self.assertEqual(empty["lengths"], 1)
        self.assertTrue(np.isfinite(empty["features"]).all())
        torch.manual_seed(42)
        model = GRUProductModel(6, 11, [3, 7], width=16, layers=2).eval()
        a = {
            k: torch.zeros((1, 8), dtype=torch.long)
            for k in ["events", "products", "gaps", "hours", "weekdays", "boundaries"]
        }
        a.update(lengths=torch.tensor([3]), features=torch.zeros(1, 2, 12))
        a["events"][:, :3] = 1
        a["products"][:, :3] = 3
        b = {k: v.clone() for k, v in a.items()}
        b["events"][:, 3:] = 5
        b["products"][:, 3:] = 7
        with torch.inference_mode():
            np.testing.assert_allclose(
                model(**a).numpy(), model(**b).numpy(), atol=1e-6
            )

    def test_preparation_uses_explicit_data_and_class_order(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            data = root / "data"
            data.mkdir()
            pd.DataFrame(
                dict(
                    encoded_event=[0, 1, 2],
                    event_type=["view", "click", "session_boundary"],
                    is_target_event=[False, True, False],
                )
            ).to_parquet(data / "event_dictionary.parquet")
            rows = [
                dict(
                    example_id="x",
                    client_id=7,
                    events=[dict(encoded_event=0, encoded_product=3, timestamp=1000)],
                )
            ]
            schema = pa.schema(
                [
                    ("example_id", pa.string()),
                    ("client_id", pa.int64()),
                    (
                        "events",
                        pa.list_(
                            pa.struct(
                                [
                                    ("encoded_event", pa.int64()),
                                    ("encoded_product", pa.int64()),
                                    ("timestamp", pa.int64()),
                                ]
                            )
                        ),
                    ),
                ]
            )
            pq.write_table(
                pa.Table.from_pylist(rows, schema=schema), data / "test.parquet"
            )
            store = prepare(data, root / "cache", "test", [3, 7], log=lambda _: None)
            self.assertEqual(store.ids.tolist(), ["x"])
            self.assertEqual(store.features.shape, (1, 2, 12))
            with self.assertRaisesRegex(ValueError, "Cache source/config changed"):
                prepare(data, root / "cache", "test", [3, 8], log=lambda _: None)

    def test_stack_does_not_consume_labels(self):
        from types import SimpleNamespace

        store = SimpleNamespace(
            classes=np.array([3, 7]),
            features=np.zeros((1, 2, 12)),
            labels=np.array([0]),
        )
        neural = dict(
            scores=np.array([[1.0, 2.0]], np.float32),
            users=np.zeros((1, 32)),
            candidates=np.zeros((1, 2, 32)),
        )
        before = stack_frame(store, np.array([0]), neural, np.zeros((1, 2)), "scores")
        store.labels[:] = 1
        pd.testing.assert_frame_equal(
            before,
            stack_frame(store, np.array([0]), neural, np.zeros((1, 2)), "scores"),
        )
        self.assertEqual(before.shape, (2, 22))

    def test_bias_regularization(self):
        z = np.zeros((20, 3))
        y = np.zeros(20, dtype=int)
        a = fit_bias(z, y, 0.1)
        b = fit_bias(z, y, 10.0)
        self.assertGreater(a[0], a[1])
        self.assertLess(np.linalg.norm(b), np.linalg.norm(a))


if __name__ == "__main__":
    unittest.main()
