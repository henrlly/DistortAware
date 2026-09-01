import unittest

import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "patchhead"))
from metrics import multiclass_accuracy, roc_auc, target_fpr_threshold, threshold_metrics  # noqa: E402


class PatchHeadMetricsTests(unittest.TestCase):
    def test_metrics_are_flat_and_count_consistent(self):
        y = np.array([[0], [0], [1], [1]])
        scores = np.array([[.1], [.9], [.8], [.2]])
        result = threshold_metrics(y, scores, .5)
        self.assertEqual(result["tn"], 1)
        self.assertEqual(result["fp"], 1)
        self.assertEqual(result["tp"], 1)
        self.assertEqual(result["fn"], 1)
        self.assertAlmostEqual(result["accuracy"], .5)

    def test_auc_and_target_fpr(self):
        y = [0, 0, 1, 1]
        scores = [.1, .2, .8, .9]
        self.assertAlmostEqual(roc_auc(y, scores), 1.0)
        threshold = target_fpr_threshold(y, scores, .5)
        self.assertEqual(threshold, .1)

    def test_three_class_metrics(self):
        result = multiclass_accuracy([0, 1, 2, 2], [0, 1, 1, 2])
        self.assertAlmostEqual(result["accuracy"], .75)
        self.assertEqual(result["tampered_n"], 2)
        self.assertAlmostEqual(result["tampered_accuracy"], .5)


if __name__ == "__main__":
    unittest.main()
