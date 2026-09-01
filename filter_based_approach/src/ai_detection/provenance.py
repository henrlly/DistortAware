"""Best-effort local provenance inspection.

Cryptographic C2PA verification should be performed with c2patool when installed;
this module only reports metadata visible to Python and never calls missing
provenance "authentic".
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image


def inspect_provenance(path: str | Path | None = None, image: Image.Image | None = None) -> dict[str, Any]:
    if image is None and path is not None:
        image = Image.open(path)
    if image is None:
        raise ValueError("provide path or image")

    exif = image.getexif()
    software = exif.get(305)  # EXIF Software tag
    return {
        "format": image.format,
        "c2pa": "not_checked_locally",
        "synthid": "not_checked_locally",
        "exif_present": bool(exif),
        "exif_software": str(software) if software else None,
        "conclusion": "unknown_without_verified_credential",
    }

