from __future__ import annotations

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]


class DetectorIsolationTests(unittest.TestCase):
    def test_detector_packages_do_not_import_other_detectors(self) -> None:
        forbidden = {
            "did": ("patchhead", "physics_engine", "filter_based_approach"),
            "patchhead": ("did", "physics_engine", "filter_based_approach"),
            "filter_based_approach": ("did", "patchhead", "physics_engine"),
            "physics/src/physics_engine": ("did", "patchhead", "filter_based_approach"),
        }
        for package, names in forbidden.items():
            package_root = ROOT / package
            for path in package_root.rglob("*.py"):
                if "tests" in path.parts or "__pycache__" in path.parts:
                    continue
                for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                    stripped = line.strip()
                    if stripped.startswith("#"):
                        continue
                    for name in names:
                        pattern = rf"^(?:from|import)\s+{re.escape(name)}(?:\.|\s|$)"
                        self.assertIsNone(
                            re.match(pattern, stripped),
                            f"{path.relative_to(ROOT)}:{line_number} imports {name}",
                        )


if __name__ == "__main__":
    unittest.main()
