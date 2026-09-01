"""Build a deterministic, loadable PrismGuard MV3 extension archive."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
EXTENSION_ROOT = ROOT / "browser_extension"
EXCLUDED_PARTS = {"tests", "demo", "wildfake_demo", "__pycache__"}
EXCLUDED_NAMES = {"README.md", ".DS_Store"}


def package_extension(output: Path) -> tuple[str, int]:
    files = [
        path
        for path in sorted(EXTENSION_ROOT.rglob("*"))
        if path.is_file()
        and path.name not in EXCLUDED_NAMES
        and not EXCLUDED_PARTS.intersection(path.relative_to(EXTENSION_ROOT).parts)
    ]
    if not files or EXTENSION_ROOT / "manifest.json" not in files:
        raise RuntimeError("extension package is missing manifest.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            relative = path.relative_to(EXTENSION_ROOT).as_posix()
            info = zipfile.ZipInfo(relative, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())
    payload = output.read_bytes()
    return hashlib.sha256(payload).hexdigest(), len(payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "dist" / "prismguard-browser-extension.zip",
    )
    args = parser.parse_args()
    digest, byte_count = package_extension(args.output.resolve())
    print(f"extension={args.output.resolve()}")
    print(f"sha256={digest}")
    print(f"bytes={byte_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
