"""Shared helpers for standalone output-viewer ZIP bundles."""

from __future__ import annotations

import json
import zipfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from .assets import COMMON_CSS

SCHEMA = "scripps.standalone_viewer_input.v1"


def _safe_script_json(obj: dict[str, Any]) -> str:
    """JSON safe to embed inside a ``<script type=application/json>`` tag."""

    return json.dumps(obj, ensure_ascii=False).replace("</", "<\\/")


def _public_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Payload copy with bulky embedded data removed for viewer_input.json."""

    return {k: v for k, v in payload.items() if k != "xyz_text"}


def write_bundle(
    *,
    output_path: Path,
    viewer_kind: str,
    index_html: str,
    viewer_js: str,
    xyz_text: str,
    xyz_file_name: str,
    title: str | None,
    source_path: str | None,
    source_label: str | None,
    smiles: str | None,
    manifest_path: str | None,
    selected_index: int | None,
    n_frames: int | None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write a standalone viewer ZIP and return its path."""

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    payload = {
        "schema": SCHEMA,
        "viewer": viewer_kind,
        "title": title,
        "source_path": source_path,
        "source_label": source_label,
        "manifest_path": manifest_path,
        "smiles": smiles,
        "selected_index": selected_index,
        "n_frames": n_frames,
        "metadata": metadata or {},
        "file_name": xyz_file_name,
        "xyz_text": xyz_text,
    }
    html = index_html.replace("__VIEWER_INPUT__", _safe_script_json(payload))

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("index.html", html)
        zf.writestr("js/viewer.js", viewer_js)
        zf.writestr("css/styles.css", COMMON_CSS)
        zf.writestr(f"data/{xyz_file_name}", xyz_text)
        zf.writestr(
            "viewer_input.json",
            json.dumps(_public_payload(payload), indent=2, ensure_ascii=False) + "\n",
        )
        zf.writestr("README.md", _readme(viewer_kind=viewer_kind, xyz_file_name=xyz_file_name))

    return output_path


def _readme(*, viewer_kind: str, xyz_file_name: str) -> str:
    title = "Conformer Ensemble Viewer" if viewer_kind == "ensemble_viewer" else "Geometry Viewer"
    return f"""# {title} Bundle

Open `index.html` in a web browser to view the structure.

The XYZ payload is embedded directly in `index.html`, so the page does not need
to fetch workflow-GUI resources, `opaat`, or HPC paths. A copy of the original
XYZ data is also included at `data/{xyz_file_name}` for inspection.

Included files:

- `index.html` — standalone viewer with embedded XYZ payload
- `js/viewer.js` — local viewer logic
- `css/styles.css` — local styling
- `data/{xyz_file_name}` — extracted XYZ artifact
- `viewer_input.json` — metadata copy with the bulky XYZ text omitted
- `README.md` — this file

The viewer loads 3Dmol.js from `https://3Dmol.org`, so internet access is
required unless the browser has already cached that script.
"""
