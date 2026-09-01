from __future__ import annotations

import unittest

import numpy as np

from harness.patchhead_robustness import (
    _compare_outputs,
    _cosine_rows,
    _pearson,
)


class CheckpointRobustnessMetricTests(unittest.TestCase):
    def test_identical_outputs_have_perfect_stability(self) -> None:
        outputs = {
            "scores": np.asarray([0.2, 0.8]),
            "patch_scores": np.asarray(
                [[[0.1, 0.2], [0.3, 0.4]], [[0.9, 0.8], [0.7, 0.6]]]
            ),
            "features": np.arange(32, dtype=np.float32).reshape(2, 2, 2, 4),
        }

        result = _compare_outputs(outputs, outputs, threshold=0.5)

        self.assertEqual(result["verdict_flips"], 0)
        self.assertEqual(result["score_absolute_drift"]["maximum"], 0.0)
        self.assertAlmostEqual(result["patch_score_pearson"]["mean"], 1.0)
        self.assertAlmostEqual(result["dense_token_cosine"]["mean"], 1.0)

    def test_grid_metrics_handle_zero_and_constant_vectors(self) -> None:
        self.assertEqual(_pearson(np.ones(4), np.ones(4)), 1.0)
        self.assertEqual(_pearson(np.ones(4), np.zeros(4)), 0.0)
        cosine = _cosine_rows(np.zeros((1, 4)), np.zeros((1, 4)))
        self.assertEqual(float(cosine[0]), 1.0)

    def test_verdict_flip_and_drift_are_reported(self) -> None:
        baseline = {
            "scores": np.asarray([0.4]),
            "patch_scores": np.asarray([[[0.2, 0.8], [0.4, 0.6]]]),
            "features": np.ones((1, 2, 2, 3)),
        }
        transformed = {
            "scores": np.asarray([0.7]),
            "patch_scores": np.asarray([[[0.8, 0.2], [0.6, 0.4]]]),
            "features": -np.ones((1, 2, 2, 3)),
        }

        result = _compare_outputs(baseline, transformed, threshold=0.5)

        self.assertEqual(result["verdict_flips"], 1)
        self.assertAlmostEqual(result["score_absolute_drift"]["maximum"], 0.3)
        self.assertAlmostEqual(result["dense_token_cosine"]["mean"], -1.0)


if __name__ == "__main__":
    unittest.main()
