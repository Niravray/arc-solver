import sys
import unittest
from pathlib import Path


LAYERS = Path(__file__).resolve().parents[1] / "src" / "layers"
sys.path.insert(0, str(LAYERS))

from graph_difference import compare_relationship_graphs  # noqa: E402
from relationship_graph import build_relationship_graph  # noqa: E402
from test_relationship_graph import make_object  # noqa: E402


class GraphDifferenceTests(unittest.TestCase):
    def test_matches_nodes_globally_and_records_movement(self):
        input_objects = [
            make_object(0, 1, [(0, 0)]),
            make_object(1, 2, [(4, 0), (4, 1)]),
        ]
        output_objects = [
            make_object(8, 1, [(3, 4)]),
            make_object(9, 2, [(1, 0), (1, 1)]),
        ]
        input_graph = build_relationship_graph(input_objects, (5, 5))
        output_graph = build_relationship_graph(output_objects, (5, 5))

        difference = compare_relationship_graphs(
            input_graph,
            output_graph,
            strategy="synthetic",
        )

        matches = {
            match["input_id"]: match for match in difference["node_matches"]
        }
        self.assertEqual(matches[0]["output_id"], 8)
        self.assertEqual(matches[1]["output_id"], 9)
        self.assertTrue(matches[0]["changes"]["moved"])
        self.assertIn("NODES_MOVED", difference["signature"])
        self.assertIn("RELATIONSHIPS_CHANGED", difference["signature"])

    def test_allows_added_and_removed_nodes(self):
        input_graph = build_relationship_graph(
            [
                make_object(0, 1, [(0, 0)]),
                make_object(1, 2, [(2, 2), (2, 3)]),
            ],
            (4, 4),
        )
        output_graph = build_relationship_graph(
            [
                make_object(4, 1, [(0, 0)]),
                make_object(5, 3, [(3, 3), (3, 2), (2, 3)]),
            ],
            (4, 4),
        )

        difference = compare_relationship_graphs(input_graph, output_graph)

        self.assertEqual(
            [(match["input_id"], match["output_id"])
             for match in difference["node_matches"]],
            [(0, 4)],
        )
        self.assertEqual(difference["removed_input_ids"], [1])
        self.assertEqual(difference["added_output_ids"], [5])
        self.assertIn("NODES_REMOVED", difference["signature"])
        self.assertIn("NODES_ADDED", difference["signature"])


if __name__ == "__main__":
    unittest.main()
