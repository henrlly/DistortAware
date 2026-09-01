from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest

from scripts.launch_browser_demo import DemoConfig, service_commands, validate_config


class LaunchBrowserDemoTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.checkpoint = self.root / "patchhead.pt"
        self.backbone = self.root / "model.safetensors"
        self.checkpoint.write_bytes(b"sealed-head")
        self.backbone.write_bytes(b"sealed-backbone")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def config(self, **overrides: object) -> DemoConfig:
        values: dict[str, object] = {
            "checkpoint": self.checkpoint,
            "checkpoint_sha256": self.digest(self.checkpoint),
            "backbone_safetensors": self.backbone,
            "backbone_sha256": self.digest(self.backbone),
            "hf_home": self.root,
            "patchhead_token": "head-token",
            "prismguard_token": "prism-token",
            "fixture_token": "fixture-token",
        }
        values.update(overrides)
        return DemoConfig(**values)  # type: ignore[arg-type]

    def test_hash_mismatch_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "checkpoint SHA-256 mismatch"):
            validate_config(self.config(checkpoint_sha256="0" * 64))

    def test_partial_prismguard_tuple_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires bundle, checkpoint"):
            validate_config(self.config(prismguard_bundle=self.checkpoint))

    def test_duplicate_ports_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "ports must be distinct"):
            validate_config(self.config(fixture_port=8765))

    def test_commands_are_loopback_only_and_disable_diagnostics(self) -> None:
        config = validate_config(self.config())
        commands = dict(service_commands(config))
        self.assertEqual(set(commands), {"PatchHead", "Wiring fixture", "Demo wall"})
        patchhead = commands["PatchHead"]
        self.assertIn("--physics-profile", patchhead)
        self.assertEqual(patchhead[patchhead.index("--physics-profile") + 1], "off")
        self.assertEqual(patchhead[patchhead.index("--artifact-profile") + 1], "off")
        wall = commands["Demo wall"]
        self.assertEqual(wall[wall.index("--bind") + 1], "127.0.0.1")

    def test_complete_prismguard_tuple_adds_pure_dino_service(self) -> None:
        bundle = self.root / "bundle.json"
        ledger = self.root / "ledger.json"
        bundle.write_text("{}", encoding="utf-8")
        ledger.write_text("{}", encoding="utf-8")
        config = validate_config(
            self.config(
                prismguard_bundle=bundle,
                prismguard_checkpoint=self.backbone,
                prismguard_license_ledger=ledger,
                prismguard_root=self.root,
            )
        )
        command = dict(service_commands(config))["PrismGuard"]
        self.assertIn("--prismguard-bundle", command)
        self.assertIn("--prismguard-license-ledger", command)
        self.assertEqual(command[command.index("--physics-profile") + 1], "off")
        self.assertEqual(command[command.index("--artifact-profile") + 1], "off")


if __name__ == "__main__":
    unittest.main()
