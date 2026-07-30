import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "layers"))
sys.path.insert(0, str(ROOT / "src" / "primitives"))

import synthesis  # noqa: E402
from synthesis import (  # noqa: E402
    DRAW,
    EXTRACT,
    MOVE,
    RECOLOR,
    REMOVE,
    SCENE_INTERPRETATIONS,
    Selector,
    build_atom_catalog,
    extract_panels,
    find_separator_lines,
    hole_count,
    induce_color_exprs,
    induce_decision_list,
    induce_selectors,
    observe_draw,
    observe_extract,
    observe_recolor,
    observe_remove,
    scene_objects,
    selector_cost,
    set_scene_interpretation,
    synthesize,
    synthesize_program,
)
from object_extractor import get_background_color  # noqa: E402


def grid(rows):
    """Build a grid from strings of digits, for readability in tests."""
    return np.array([[int(c) for c in row] for row in rows], dtype=np.int32)


def task(pairs, test_input=None):
    return {
        "train": [{"input": grid(i), "output": grid(o)} for i, o in pairs],
        "test": [{"input": grid(test_input or pairs[0][0])}],
    }


class SceneReadingTests(unittest.TestCase):
    """Object extraction is a CHOICE; these pin what each choice yields."""

    def setUp(self):
        set_scene_interpretation({"connectivity": 4})

    def tearDown(self):
        set_scene_interpretation(SCENE_INTERPRETATIONS[0][1])

    def test_connectivity_changes_what_counts_as_one_object(self):
        # Two diagonally touching pixels: separate under 4, joined under 8.
        g = grid(["1000", "0100", "0000", "0000"])
        set_scene_interpretation({"connectivity": 4})
        self.assertEqual(len(scene_objects(g, 0)), 2)
        set_scene_interpretation({"connectivity": 8})
        self.assertEqual(len(scene_objects(g, 0)), 1)

    def test_separator_lines_are_found_on_both_axes(self):
        # Panels must have internal variation, otherwise their own columns
        # read as uniform and every column looks like a divider -- see
        # test_uniform_panels_defeat_separator_detection below.
        g = grid(["10530", "01503", "55555", "10530"])
        rows, cols = find_separator_lines(g, 0)
        self.assertEqual(rows, [2])
        self.assertEqual(cols, [2])

    def test_panels_split_a_separator_grid(self):
        g = grid(["10530", "01503", "10530"])
        panels = extract_panels(g, 0)
        self.assertEqual(len(panels), 2)
        self.assertEqual({p["width"] for p in panels}, {2})
        # a panel's colour is its dominant non-background colour
        self.assertEqual({p["color"] for p in panels}, {1, 3})

    def test_uniform_panels_defeat_separator_detection(self):
        """
        KNOWN LIMITATION, pinned so it is not mistaken for a regression.

        find_separator_lines flags any uniform non-background line, but a
        solid-coloured panel's own columns are uniform too -- so a grid of
        solid blocks has every column flagged, no segment survives, and no
        panel is produced. Found while writing these tests. Separator
        detection needs to distinguish a divider from a panel interior
        (e.g. by requiring the line to span the full grid AND its colour
        to appear nowhere else) before panels can handle solid-block
        layouts.
        """
        g = grid(["11522", "11522"])
        _rows, cols = find_separator_lines(g, 0)
        self.assertEqual(cols, [0, 1, 2, 3, 4])
        self.assertEqual(extract_panels(g, 0), [])

    def test_panels_returns_nothing_without_separators(self):
        self.assertEqual(extract_panels(grid(["11", "22"]), 0), [])

    def test_hole_count_counts_only_enclosed_background(self):
        ring = {"grid": grid(["111", "101", "111"]), "color": 1}
        self.assertEqual(hole_count(ring), 1)
        open_shape = {"grid": grid(["111", "100", "111"]), "color": 1}
        self.assertEqual(hole_count(open_shape), 0)


class SelectorInductionTests(unittest.TestCase):
    """The 'which objects?' hole, shared by every operator."""

    def _observed(self, g, positive_ids):
        objs = scene_objects(g, 0)
        pos = [(o, None) for o in objs if o["id"] in positive_ids]
        neg = [o for o in objs if o["id"] not in positive_ids]
        return [(objs, pos, neg, 0, g)]

    def test_finds_a_single_atom_when_one_separates(self):
        g = grid(["00000", "01100", "01100", "00002", "00000"])
        objs = scene_objects(g, 0)
        biggest = max(objs, key=lambda o: o["pixel_count"])
        found = induce_selectors(self._observed(g, {biggest["id"]}))
        self.assertTrue(found)
        self.assertIn("is_largest", [str(s) for s in found])

    def test_returns_nothing_when_objects_are_indistinguishable(self):
        # Two identical single pixels, both away from every border, so not
        # even a position atom separates them. (Pixels at opposite ENDS
        # would be separable by touches_left, which is correct behaviour.)
        g = grid(["00000", "01010", "00000"])
        objs = scene_objects(g, 0)
        self.assertEqual(len(objs), 2)
        found = induce_selectors(self._observed(g, {objs[0]["id"]}))
        self.assertEqual([str(s) for s in found], [])

    def test_structural_atoms_outrank_metric_ones(self):
        # This is the 72ca375d fix: with equal atom counts, a selector
        # pinned to a measurement must lose to a structural one.
        metric = Selector([("size==4", lambda o, objs: True)])
        structural = Selector([("is_largest", lambda o, objs: True)])
        self.assertLess(selector_cost(structural), selector_cost(metric))

    def test_more_evidence_never_widens_the_candidate_set(self):
        g1 = grid(["00000", "01100", "00002", "00000"])
        g2 = grid(["00000", "03300", "00004", "00000"])
        objs1 = scene_objects(g1, 0)
        objs2 = scene_objects(g2, 0)
        big1 = max(objs1, key=lambda o: o["pixel_count"])
        big2 = max(objs2, key=lambda o: o["pixel_count"])
        one = [(objs1, [(big1, None)],
                [o for o in objs1 if o is not big1], 0, g1)]
        both = one + [(objs2, [(big2, None)],
                       [o for o in objs2 if o is not big2], 0, g2)]
        self.assertLessEqual(len(induce_selectors(both)),
                             len(induce_selectors(one)))


class ValueExpressionTests(unittest.TestCase):
    def _obs(self, values):
        g = grid(["000", "010", "000"])
        objs = scene_objects(g, 0)
        return [(objs[0], objs, g, 0, v) for v in values]

    def test_single_observed_value_yields_a_literal(self):
        exprs = induce_color_exprs(self._obs([7]))
        kinds = [e.kind for e, _ in exprs]
        self.assertIn("literal", kinds)

    def test_literal_is_ranked_cheaper_than_a_table(self):
        exprs = induce_color_exprs(self._obs([7]))
        by_kind = {e.kind: cost for e, cost in exprs}
        self.assertEqual(by_kind["literal"], 0)
        for kind, cost in by_kind.items():
            if kind != "literal":
                self.assertGreater(cost, 0)

    def test_geometric_decision_lists_require_symbolic_keys(self):
        # The Move overfit fix: offsets keyed on a measurement are
        # memorisation, so such a list must be refused outright.
        g = grid(["00000", "01000", "00220", "00000"])
        objs = scene_objects(g, 0)
        small = min(objs, key=lambda o: o["pixel_count"])
        big = max(objs, key=lambda o: o["pixel_count"])
        observed = [(objs, [(small, (1, 0)), (big, (2, 0))], [], 0, g)]
        result = induce_decision_list(observed)
        if result is not None:
            for selector, _ in result.clauses:
                for name, _fn in selector.atoms:
                    self.assertTrue(name.startswith("color=="))


class ObserverTests(unittest.TestCase):
    """Each observer must label its own change and reject others."""

    def test_recolor_labels_only_recoloured_objects(self):
        obs = observe_recolor(grid(["00000", "01100", "00002", "00000"]),
                              grid(["00000", "05500", "00002", "00000"]))
        self.assertIsNotNone(obs)
        _objs, positives, negatives, _bg, _g = obs
        self.assertEqual(len(positives), 1)
        self.assertEqual(positives[0][1], 5)
        self.assertEqual(len(negatives), 1)

    def test_remove_labels_the_vanished_object(self):
        obs = observe_remove(grid(["00000", "01100", "00002", "00000"]),
                             grid(["00000", "01100", "00000", "00000"]))
        self.assertIsNotNone(obs)
        _objs, positives, negatives, _bg, _g = obs
        self.assertEqual(len(positives), 1)
        self.assertEqual(len(negatives), 1)

    def test_remove_rejects_a_pair_where_something_also_changed(self):
        self.assertIsNone(
            observe_remove(grid(["00000", "01100", "00002", "00000"]),
                           grid(["00000", "05500", "00000", "00000"])))

    def test_draw_requires_a_pure_addition(self):
        self.assertIsNotNone(
            observe_draw(grid(["000", "010", "000"]),
                         grid(["000", "011", "000"])))
        # something was removed as well -> not a pure addition
        self.assertIsNone(
            observe_draw(grid(["020", "010", "000"]),
                         grid(["000", "011", "000"])))

    def test_extract_requires_a_shape_change(self):
        self.assertIsNone(
            observe_extract(grid(["010", "000"]), grid(["020", "000"])))

    def test_extract_finds_the_object_that_became_the_output(self):
        obs = observe_extract(
            grid(["00000", "01100", "01100", "00030", "00000"]),
            grid(["11", "11"]))
        self.assertIsNotNone(obs)
        _objs, positives, negatives, _bg, _g = obs
        self.assertEqual(len(positives), 1)
        self.assertTrue(negatives)


class LoopTests(unittest.TestCase):
    """The counterexample-guided loop and its guarantees."""

    def test_induces_a_rule_that_explains_every_pair(self):
        t = task([
            (["00000", "01100", "01100", "00002", "00000"],
             ["00000", "05500", "05500", "00002", "00000"]),
            (["00000", "03000", "00000", "00440", "00000"],
             ["00000", "03000", "00000", "00550", "00000"]),
        ])
        program, trace = synthesize(t, RECOLOR)
        self.assertIsNotNone(program, msg="\n".join(trace))
        for pair in t["train"]:
            got = program.apply(pair["input"],
                                get_background_color(pair["input"]))
            self.assertIsNotNone(got)
            np.testing.assert_array_equal(got, pair["output"])

    def test_returns_none_when_no_rule_fits(self):
        # Pair 2 contradicts pair 1: the same object goes to two colours.
        t = task([
            (["00000", "01100", "00000"], ["00000", "05500", "00000"]),
            (["00000", "01100", "00000"], ["00000", "07700", "00000"]),
        ])
        program, _trace = synthesize(t, RECOLOR)
        self.assertIsNone(program)

    def test_a_returned_rule_always_survives_the_final_check(self):
        # Whatever comes back must reproduce every training output; this is
        # the invariant the whole engine's precision rests on.
        t = task([
            (["00000", "01100", "00002", "00000"],
             ["00000", "05500", "00002", "00000"]),
            (["00000", "03300", "00004", "00000"],
             ["00000", "05500", "00004", "00000"]),
        ])
        program, _ = synthesize(t, RECOLOR)
        if program is not None:
            for pair in t["train"]:
                got = program.apply(pair["input"],
                                    get_background_color(pair["input"]))
                np.testing.assert_array_equal(got, pair["output"])


class InterpretationSearchTests(unittest.TestCase):
    def tearDown(self):
        set_scene_interpretation(SCENE_INTERPRETATIONS[0][1])

    def test_default_interpretation_is_restored_when_nothing_is_found(self):
        t = task([(["012", "120", "201"], ["210", "021", "102"])])
        program, _name = synthesize_program(t)
        if program is None:
            self.assertEqual(synthesis._ACTIVE_INTERPRETATION,
                             dict(SCENE_INTERPRETATIONS[0][1]))

    def test_applicability_gate_skips_panels_without_separators(self):
        plain = task([(["010", "000"], ["020", "000"])])
        self.assertFalse(synthesis._interpretation_applies("panels", plain))

    def test_applicability_gate_skips_connected8_when_it_reads_the_same(self):
        # A single solid block reads identically under both connectivities.
        same = task([(["00000", "01100", "01100", "00000"],
                      ["00000", "05500", "05500", "00000"])])
        self.assertFalse(synthesis._interpretation_applies("connected8", same))

    def test_applicability_gate_allows_connected8_for_diagonal_shapes(self):
        diagonal = task([(["1000", "0100", "0010", "0000"],
                          ["2000", "0200", "0020", "0000"])])
        self.assertTrue(synthesis._interpretation_applies("connected8", diagonal))


class DisabledCapabilityTests(unittest.TestCase):
    """
    These pin the measured-off switches. If someone flips one, the test
    fails and points them at the measurement rather than letting a silent
    precision regression through.
    """

    def test_measured_off_switches_stay_off(self):
        self.assertFalse(synthesis.USE_RELATIONAL_ATOMS)
        self.assertFalse(synthesis.USE_REGION_SPLIT)
        self.assertEqual(synthesis.MAX_CANDIDATE_RESTARTS, 1)
        self.assertEqual(synthesis.MAX_CONJUNCTION, 2)


if __name__ == "__main__":
    unittest.main()
