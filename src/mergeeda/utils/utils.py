"""Shared utilities for OpenAI multimodal content block construction."""

import base64
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

MATERIAL_TAG_PATTERN = re.compile(r"<material:([\w\-._]+)>")

IMAGE_SUFFIXES: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".gif", ".webp"})

MIME_MAP: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


def build_image_block(image_path: Path) -> dict:
    """Encode a local image file as a base64 OpenAI image_url content block."""
    mime = MIME_MAP.get(image_path.suffix.lower(), "image/jpeg")
    b64 = base64.standard_b64encode(image_path.read_bytes()).decode("utf-8")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "high"},
    }


def build_material_content_blocks(
    material_filenames: list[str],
    materials_dir: Path,
) -> list[dict]:
    """Build OpenAI content blocks for each material filename.

    Images are encoded as base64 image_url blocks.
    Tables (.txt) are returned as labeled text blocks.
    Unrecognised extensions are skipped with a warning.
    """
    blocks: list[dict] = []
    for filename in material_filenames:
        material_path = materials_dir / filename
        if not material_path.exists():
            logger.warning(f"Material not found, skipping: {material_path}")
            continue
        suffix = material_path.suffix.lower()
        if suffix in IMAGE_SUFFIXES:
            blocks.append(build_image_block(material_path))
        elif suffix == ".txt":
            table_text = material_path.read_text(encoding="utf-8")
            blocks.append(
                {"type": "text", "text": f"[Table: {filename}]\n{table_text}"}
            )
        else:
            logger.warning(f"Unsupported material type, skipping: {filename}")
    return blocks
