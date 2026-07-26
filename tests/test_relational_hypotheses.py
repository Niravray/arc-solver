import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "layers"))
sys.path.insert(0, str(ROOT / "src" / "primitives"))

from verifier import generate_hypotheses, verify_all  # noqa: E402


class RelationalHypothesisTests(unittest.TestCase):
    def test_generates_and_verifies_move_until_touching(self):
        task = {
            "id": "synthetic_move_until_touching",
            "train": [
                {
                    "input": np.array([[1, 0, 0, 0, 2]]),
                    "output": np.array([[0, 0, 0, 1, 2]]),
                },
                {
                    "input": np.array([[3, 0, 0, 0, 0, 4]]),
                    "output": np.array([[0, 0, 0, 0, 3, 4]]),
                },
            ],
            "test": [{"input": np.array([[5, 0, 0, 0, 6]])}],
        }

        hypotheses = generate_hypotheses(task)
        relational = [
            hypothesis
            for hypothesis in hypotheses
            if "until_touching" in hypothesis.name
        ]
        verified = verify_all(relational, task)

        self.assertGreater(len(relational), 0)
        self.assertTrue(
            any(
                hypothesis.name.startswith(
                    "move_leftmost_object_until_touching_rightmost_object"
                )
                for hypothesis in verified
            )
        )


if __name__ == "__main__":
    unittest.main()
