import sys
import unittest
from pathlib import Path

import numpy as np


LAYERS = Path(__file__).resolve().parents[1] / "src" / "layers"
sys.path.insert(0, str(LAYERS))

from object_extractor import extract_objects_multi_strategy  # noqa: E402
from relationship_graph import (  # noqa: E402
    GraphValidationError,
    build_relationship_graph,
    build_relationship_graph_interpretations,
    find_largest_object,
    find_smallest_object,
    validate_relationship_graph,
)


def make_object(object_id, color, pixels):
    rows = [row for row, _ in pixels]
    cols = [col for _, col in pixels]
    min_row, max_row = min(rows), max(rows)
    min_col, max_col = min(cols), max(cols)
    height = max_row - min_row + 1
    width = max_col - min_col + 1
    mask = [0] * (height * width)
    for row, col in pixels:
        mask[(row - min_row) * width + col - min_col] = 1

    return {
        "id": object_id,
        "color": color,
        "pixels": pixels,
        "pixel_count": len(pixels),
        "min_row": min_row,
        "max_row": max_row,
        "min_col": min_col,
        "max_col": max_col,
        "height": height,
        "width": width,
        "centroid_row": sum(rows) / len(rows),
        "centroid_col": sum(cols) / len(cols),
        "fingerprint": tuple(mask),
    }


class RelationshipGraphTests(unittest.TestCase):
    def test_direction_describes_b_relative_to_a(self):
        top = make_object(0, 1, [(0, 2)])
        bottom = make_object(1, 2, [(4, 2)])

        graph = build_relationship_graph([top, bottom], (5, 5))

        self.assertEqual(
            graph["pairs"][(top["id"], bottom["id"])]["direction_a_to_b"],
            "down",
        )
        self.assertEqual(
            graph["pairs"][(bottom["id"], top["id"])]["direction_a_to_b"],
            "up",
        )

    def test_reciprocal_relations_validate(self):
        left = make_object(0, 1, [(2, 0), (2, 1)])
        right = make_object(1, 1, [(2, 3), (2, 4)])

        graph = build_relationship_graph([left, right], (5, 5))

        self.assertIs(validate_relationship_graph(graph), graph)
        self.assertTrue(graph["pairs"][(0, 1)]["a_left_of_b"])
        self.assertTrue(graph["pairs"][(1, 0)]["a_right_of_b"])
        self.assertTrue(graph["pairs"][(0, 1)]["same_color"])

    def test_size_ties_use_lowest_id(self):
        later = make_object(7, 1, [(3, 3)])
        earlier = make_object(2, 2, [(0, 0)])

        graph = build_relationship_graph([later, earlier], (4, 4))

        self.assertEqual(graph["size_ranking"], [2, 7])
        self.assertEqual(find_largest_object(graph)["id"], 2)
        self.assertEqual(find_smallest_object(graph)["id"], 2)

    def test_validation_rejects_missing_reverse_edge(self):
        first = make_object(0, 1, [(0, 0)])
        second = make_object(1, 2, [(1, 1)])
        graph = build_relationship_graph([first, second], (2, 2))
        del graph["pairs"][(1, 0)]

        with self.assertRaisesRegex(GraphValidationError, "Pair set mismatch"):
            validate_relationship_graph(graph)

    def test_no_background_strategy_really_keeps_background(self):
        grid = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ])

        strategies = extract_objects_multi_strategy(grid)

        self.assertEqual(len(strategies["connected_4"]), 1)
        self.assertEqual(
            {obj["color"] for obj in strategies["no_background"]},
            {0, 1},
        )

    def test_builds_a_valid_graph_for_every_interpretation(self):
        grid = np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ])

        interpretations = build_relationship_graph_interpretations(grid)

        self.assertEqual(
            set(interpretations["graphs"]),
            {"connected_4", "connected_8", "by_color", "no_background"},
        )
        self.assertEqual(interpretations["skipped"], {})
        self.assertGreater(
            len(interpretations["graphs"]["connected_4"]["objects"]),
            len(interpretations["graphs"]["connected_8"]["objects"]),
        )
        for graph in interpretations["graphs"].values():
            self.assertIs(validate_relationship_graph(graph), graph)

    def test_pathological_interpretations_are_bounded(self):
        grid = np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ])

        interpretations = build_relationship_graph_interpretations(
            grid,
            max_objects=1,
        )

        self.assertIn("connected_4", interpretations["skipped"])
        self.assertIn("no_background", interpretations["skipped"])
        self.assertIn("connected_8", interpretations["graphs"])

    def test_exact_geometry_and_diagonal_contact_are_preserved(self):
        upper_left = make_object(0, 1, [(0, 0)])
        lower_right = make_object(1, 2, [(1, 1)])

        graph = build_relationship_graph([upper_left, lower_right], (3, 3))
        relation = graph["pairs"][(0, 1)]

        self.assertEqual(relation["delta_row"], 1.0)
        self.assertEqual(relation["delta_col"], 1.0)
        self.assertEqual(relation["direction_8way_a_to_b"], "down_right")
        self.assertFalse(relation["touching"])
        self.assertTrue(relation["touching_diagonally"])
        self.assertTrue(relation["touching_8"])
        self.assertEqual(relation["vertical_gap"], 0)
        self.assertEqual(relation["horizontal_gap"], 0)

    def test_contact_sides_and_shared_boundary_are_preserved(self):
        left = make_object(0, 1, [(0, 0), (1, 0)])
        right = make_object(1, 2, [(0, 1), (1, 1)])

        graph = build_relationship_graph([left, right], (2, 2))
        forward = graph["pairs"][(0, 1)]
        reverse = graph["pairs"][(1, 0)]

        self.assertEqual(forward["contact_sides_a_to_b"], ("right",))
        self.assertEqual(reverse["contact_sides_a_to_b"], ("left",))
        self.assertEqual(forward["shared_edge_count"], 2)

    def test_shape_relation_handles_dimensions_and_rotation(self):
        horizontal = make_object(0, 1, [(0, 0), (0, 1), (0, 2)])
        vertical = make_object(1, 2, [(0, 4), (1, 4), (2, 4)])

        graph = build_relationship_graph([horizontal, vertical], (3, 5))
        relation = graph["pairs"][(0, 1)]

        self.assertFalse(relation["same_shape"])
        self.assertTrue(relation["same_shape_up_to_transform"])
        self.assertEqual(relation["shape_relation"], "rotated")


if __name__ == "__main__":
    unittest.main()
