from __future__ import annotations

import unittest

import numpy as np

from physics_engine.geometry import (
    fit_line_bundle,
    fit_multiple_line_bundles,
    homogeneous_point_payload,
)


class LineBundleTests(unittest.TestCase):
    def test_fits_finite_vanishing_point_with_one_outlier(self) -> None:
        vanishing_point = np.asarray([12.0, -4.0])
        anchors = [
            np.asarray([-8.0, -9.0]),
            np.asarray([-6.0, 7.0]),
            np.asarray([0.0, 10.0]),
            np.asarray([7.0, 8.0]),
            np.asarray([9.0, -10.0]),
        ]
        segments = []
        for anchor in anchors:
            direction = vanishing_point - anchor
            first = anchor + 0.12 * direction
            second = anchor + 0.55 * direction
            segments.append((first[0], first[1], second[0], second[1]))
        segments.append((-7.0, 1.0, -2.0, 8.0))

        fit = fit_line_bundle(
            segments,
            weights=[1.0] * len(segments),
            threshold_degrees=1.0,
            min_inliers=4,
        )

        self.assertIsNotNone(fit)
        assert fit is not None
        payload = homogeneous_point_payload(fit.point)
        self.assertEqual(payload["kind"], "finite")
        self.assertTrue(np.allclose(payload["xy"], vanishing_point, atol=1e-5))
        self.assertEqual(int(np.count_nonzero(fit.inliers)), 5)
        self.assertAlmostEqual(fit.support_fraction, 5.0 / 6.0)

    def test_fits_parallel_family_as_point_at_infinity(self) -> None:
        segments = [
            (-10.0, -4.0, 10.0, 6.0),
            (-10.0, -1.0, 10.0, 9.0),
            (-10.0, 3.0, 10.0, 13.0),
            (-10.0, 7.0, 10.0, 17.0),
        ]

        fit = fit_line_bundle(
            segments,
            weights=[1.0] * len(segments),
            threshold_degrees=0.25,
            min_inliers=3,
        )

        self.assertIsNotNone(fit)
        assert fit is not None
        payload = homogeneous_point_payload(fit.point)
        self.assertEqual(payload["kind"], "infinite")
        direction = np.asarray(payload["direction"])
        expected = np.asarray([2.0, 1.0]) / np.sqrt(5.0)
        self.assertAlmostEqual(abs(float(np.dot(direction, expected))), 1.0, places=6)
        self.assertEqual(int(np.count_nonzero(fit.inliers)), 4)

    def test_multiple_bundle_fit_labels_two_directions(self) -> None:
        horizontal = [(0.0, float(y), 20.0, float(y)) for y in range(0, 20, 4)]
        vertical = [(float(x), 0.0, float(x), 20.0) for x in range(1, 21, 4)]
        segments = horizontal + vertical

        fit = fit_multiple_line_bundles(
            segments,
            weights=[1.0] * len(segments),
            max_bundles=3,
            threshold_degrees=0.25,
            min_inliers=4,
            min_global_support_fraction=0.1,
        )

        self.assertEqual(len(fit.bundles), 2)
        self.assertTrue(np.all(fit.assignments >= 0))
        self.assertEqual(len(set(fit.assignments[:5].tolist())), 1)
        self.assertEqual(len(set(fit.assignments[5:].tolist())), 1)
        self.assertNotEqual(int(fit.assignments[0]), int(fit.assignments[5]))


if __name__ == "__main__":
    unittest.main()
