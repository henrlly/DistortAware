from __future__ import annotations

import unittest

from physics_engine.annotations import ReflectionPair, ShadowPair
from physics_engine.reflection import analyze_reflections
from physics_engine.shadow import analyze_cast_shadows


class CastShadowTests(unittest.TestCase):
    def test_parallel_shadow_vectors_are_consistent(self) -> None:
        pairs = [
            ShadowPair((100.0, 100.0), (155.0, 135.0)),
            ShadowPair((180.0, 140.0), (235.0, 175.0)),
            ShadowPair((260.0, 90.0), (315.0, 125.0)),
            ShadowPair((330.0, 180.0), (385.0, 215.0)),
        ]

        result = analyze_cast_shadows(pairs, width=500, height=300)

        self.assertTrue(result.applicable)
        self.assertEqual(result.status, "consistent")
        self.assertIsNotNone(result.violation_score)
        assert result.violation_score is not None
        self.assertLess(result.violation_score, 0.1)
        self.assertEqual(
            result.measurements["estimated_projected_light"]["kind"], "infinite"
        )

    def test_incompatible_shadow_vectors_are_inconsistent(self) -> None:
        pairs = [
            ShadowPair((50.0, 50.0), (130.0, 52.0)),
            ShadowPair((200.0, 40.0), (204.0, 125.0)),
            ShadowPair((70.0, 210.0), (135.0, 145.0)),
            ShadowPair((260.0, 190.0), (195.0, 120.0)),
        ]

        result = analyze_cast_shadows(pairs, width=400, height=300)

        self.assertTrue(result.applicable)
        self.assertEqual(result.status, "inconsistent")
        self.assertGreaterEqual(result.violation_score or 0.0, 0.62)

    def test_too_few_pairs_are_not_applicable(self) -> None:
        result = analyze_cast_shadows(
            [ShadowPair((10.0, 10.0), (30.0, 20.0))], width=100, height=100
        )

        self.assertFalse(result.applicable)
        self.assertEqual(result.status, "not_applicable")
        self.assertIsNone(result.violation_score)


class ReflectionTests(unittest.TestCase):
    def test_parallel_connectors_are_consistent(self) -> None:
        pairs = [
            ReflectionPair((80.0, 40.0), (220.0, 40.0)),
            ReflectionPair((75.0, 85.0), (225.0, 85.0)),
            ReflectionPair((70.0, 130.0), (230.0, 130.0)),
            ReflectionPair((65.0, 175.0), (235.0, 175.0)),
        ]

        result = analyze_reflections(pairs, width=300, height=220)

        self.assertTrue(result.applicable)
        self.assertEqual(result.status, "consistent")
        self.assertIsNotNone(result.violation_score)
        assert result.violation_score is not None
        self.assertLess(result.violation_score, 0.1)
        self.assertEqual(
            result.measurements["reflection_vanishing_point"]["kind"], "infinite"
        )

    def test_incompatible_connectors_are_inconsistent(self) -> None:
        pairs = [
            ReflectionPair((30.0, 30.0), (160.0, 32.0)),
            ReflectionPair((40.0, 80.0), (42.0, 190.0)),
            ReflectionPair((80.0, 190.0), (175.0, 105.0)),
            ReflectionPair((210.0, 170.0), (140.0, 85.0)),
        ]

        result = analyze_reflections(pairs, width=300, height=220)

        self.assertTrue(result.applicable)
        self.assertEqual(result.status, "inconsistent")
        self.assertGreaterEqual(result.violation_score or 0.0, 0.62)


if __name__ == "__main__":
    unittest.main()
