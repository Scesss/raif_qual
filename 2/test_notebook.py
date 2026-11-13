"""Exercise notebook parsing and async ordering with a fake client, without API calls."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
import unittest


class NotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        notebook = json.loads(
            Path(__file__).with_name("raif_fcs_2_final.ipynb").read_text()
        )
        cls.namespace = {}
        for cell in notebook["cells"]:
            source = "".join(cell.get("source", []))
            if cell["cell_type"] == "code" and any(
                marker in source
                for marker in [
                    "def text2label(",
                    "MODEL_NAME =",
                    "def response2label(",
                    "def classify_sentiment(",
                ]
            ):
                exec(compile(source, "notebook_cell", "exec"), cls.namespace)

    def test_original_model_and_generation_parameters(self):
        self.assertEqual(self.namespace["MODEL_NAME"], "mistral-large-4")
        self.assertEqual(
            self.namespace["SAMPLING_PARAMETERS"],
            dict(temperature=0.1, top_p=1.0, reasoning_effort="low", max_tokens=4096),
        )
        self.assertEqual(self.namespace["MAX_CONCURRENT"], 3)

    def test_label_parser_rejects_unexpected_output(self):
        parse = self.namespace["response2label"]
        self.assertEqual(parse(" Positive \n"), 1)
        self.assertEqual(parse("negative"), 0)
        with self.assertRaises(ValueError):
            parse("probably positive")
        with self.assertRaises(TypeError):
            parse(None)

    def test_batch_preserves_input_order_with_fake_client(self):
        namespace = self.namespace

        async def main():
            active = 0
            maximum = 0

            async def fake_complete(**kwargs):
                nonlocal active, maximum
                active += 1
                maximum = max(maximum, active)
                text = kwargs["messages"][-1]["content"]
                await asyncio.sleep(0.001 if text == "slow" else 0)
                active -= 1
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(
                                content="negative" if text == "slow" else "positive"
                            )
                        )
                    ]
                )

            namespace["llm_client"] = SimpleNamespace(
                chat=SimpleNamespace(complete_async=fake_complete)
            )
            namespace["REQUEST_DELAY"] = 0
            namespace["request_lock"] = asyncio.Lock()
            namespace["next_request_start"] = 0.0
            self.assertEqual(
                await namespace["classify_batch"](["slow", "fast", "fast", "slow"]),
                [0, 1, 1, 0],
            )
            self.assertLessEqual(maximum, 3)

        asyncio.run(main())


if __name__ == "__main__":
    unittest.main()
