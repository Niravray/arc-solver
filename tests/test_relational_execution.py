import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "layers"))
sys.path.insert(0, str(ROOT / "src" / "primitives"))

from relational import (  # noqa: E402
    apply_relational_primitive,
    connect_aligned_objects,
    move_until_touching,
)
from relational_selectors import (  # noqa: E402
    aligned_objects,
    nearest_objects,
    related_objects,
    touching_objects,
)
from relationship_graph import build_relationship_graph  # noqa: E402
from test_relationship_graph import make_object  # noqa: E402


class RelationalExecutionTests(unittest.TestCase):
    def test_selectors_preserve_ties(self):
        anchor = make_object(0, 1, [(2, 2)])
        left = make_object(1, 2, [(2, 0)])
        right = make_object(2, 3, [(2, 4)])
        graph = build_relationship_graph([anchor, left, right], (5, 5))

        self.assertEqual(
            [obj["id"] for obj in nearest_objects(graph, 0)],
            [1, 2],
        )
        self.assertEqual(
            [obj["id"] for obj in aligned_objects(graph, 0)],
            [1, 2],
        )
        self.assertEqual(
            [obj["id"] for obj in related_objects(
                graph, 0, "same_centroid_row"
            )],
            [1, 2],
        )

    def test_touching_selector_can_include_diagonals(self):
        anchor = make_object(0, 1, [(0, 0)])
        diagonal = make_object(1, 2, [(1, 1)])
        graph = build_relationship_graph([anchor, diagonal], (2, 2))

        self.assertEqual(touching_objects(graph, 0), [])
        self.assertEqual(
            [obj["id"] for obj in touching_objects(
                graph, 0, include_diagonal=True
            )],
            [1],
        )

    def test_move_until_touching_preserves_target(self):
        grid = np.array([[1, 0, 0, 0, 2]])
        moving = make_object(0, 1, [(0, 0)])
        target = make_object(1, 2, [(0, 4)])

        result = move_until_touching(
            grid,
            moving,
            target,
            background=0,
            direction="right",
        )

        np.testing.assert_array_equal(
            result,
            np.array([[0, 0, 0, 1, 2]]),
        )

    def test_connect_and_generic_bridge(self):
        grid = np.array([[1, 0, 0, 0, 2]])
        first = make_object(0, 1, [(0, 0)])
        second = make_object(1, 2, [(0, 4)])
        graph = build_relationship_graph([first, second], (1, 5))

        direct = connect_aligned_objects(grid, first, second, color=3)
        bridged = apply_relational_primitive(
            "connect_aligned",
            grid,
            graph,
            0,
            1,
            color=3,
        )

        expected = np.array([[1, 3, 3, 3, 2]])
        np.testing.assert_array_equal(direct, expected)
        np.testing.assert_array_equal(bridged, expected)


if __name__ == "__main__":
    unittest.main()
