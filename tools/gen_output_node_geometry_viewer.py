#!/usr/bin/env python3
"""Generate GUI Output/Layout node ZIP — standalone single-geometry viewer.

This node is a script-backed Layout output.  Its bundle only contains a small
shim; the real implementation lives in
``scripps_workflow.output_viewers.geometry_bundle``.  The source input may be a
``wf.pointer.v1`` JSON string or a concrete XYZ path.  The script writes a
standalone ``geometry_viewer_bundle.zip`` that users can download and open
locally.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import zipfile
from pathlib import Path

OUT_DIR = Path("new_nodes_output")
PLACEHOLDER_NODE_ID = 999
NODE_NAME = "geometry_viewer"
NODE_DESCRIPTION = (
    "Create a downloadable standalone ZIP bundle containing an interactive "
    "3Dmol.js single-geometry viewer. The source input may be a wf.pointer.v1 "
    "JSON string or a concrete XYZ path."
)
NODE_VERSION = "1.2.0"
OUTPUT_ZIP = "geometry_viewer_bundle.zip"
AUTHOR = {
    "user_id": 102,
    "name": "Lucas",
    "lastname": "Abounader",
    "lab": "Shenvi",
    "email": "labounader@scripps.edu",
    "main_role": "Designer",
    "first_connection": "2026-02-13 21:09:00",
    "last_connection": "2026-05-16 14:30:00",
    "isAzureUser": 0,
}

SCRIPT_PY = """#!/usr/bin/env python3
\"\"\"GUI shim for geometry_viewer.

All real behavior lives in scripps_workflow.output_viewers.geometry_bundle.
\"\"\"

from __future__ import annotations

import sys

try:
    from scripps_workflow.output_viewers.geometry_bundle import main
except Exception as exc:  # pragma: no cover - runtime diagnostics for GUI users
    print(
        "[geometry_viewer] Could not import scripps_workflow.output_viewers."
        "geometry_bundle. Make sure scripps-workflow is installed in the HPC "
        "Python environment used by the GUI node. Original error: " + repr(exc),
        file=sys.stderr,
    )
    raise

if __name__ == "__main__":
    raise SystemExit(main())
"""


def _input(name: str, typ: str = "text", *, required: int = 0, tags: str = "") -> dict:
    return {
        "node_input_id": secrets.randbelow(900000) + 1000,
        "node_id": PLACEHOLDER_NODE_ID,
        "name": name,
        "type": typ,
        "tags": tags,
        "required": required,
        "new_id": secrets.randbelow(900000) + 1000,
    }


def build_manifest(*, block_key: str, script_path: str) -> dict:
    file_id = secrets.randbelow(900000) + 1000
    file_info = {
        "node_files_info_id": file_id,
        "node_id": PLACEHOLDER_NODE_ID,
        "file_name": "script.py",
        "file_format": "python",
        "upload_method": "code",
        "path": script_path,
    }
    layout_file = {
        **file_info,
        "screed": 0,
        "col": 0,
        "block_key": block_key,
        "entry_point": True,
        "output_file_name": OUTPUT_ZIP,
        "output_file_type": "zip",
        "screeds": 1,
        "cols": 1,
    }
    return {
        "node_id": PLACEHOLDER_NODE_ID,
        "name": NODE_NAME,
        "node_type": "Output",
        "description": NODE_DESCRIPTION,
        "category": "Layout",
        "domain": "public",
        "author": AUTHOR,
        "node_value": None,
        "directives": "#no_subfolder",
        "is_mpi": 0,
        "processing_type": "",
        "custom_value": 0,
        "file_tracking": 1,
        "limit": "",
        "files_info": [file_info],
        "inputs": [
            _input("source", "text", required=1, tags="wf.pointer.v1 or xyz path"),
            _input("smiles", "text", required=0),
            _input("title", "text", required=0),
            _input("conformer_index", "text", required=0, tags="optional: best or 1-based index"),
            _input("output_name", "text", required=0, tags="optional zip filename"),
        ],
        "outputs": [],
        "params": [],
        "version": NODE_VERSION,
        "host": "workflow.scripps.edu",
        "layout": {
            "node_id": PLACEHOLDER_NODE_ID,
            "screeds": 1,
            "cols": 1,
            "dynamic_rows": 0,
            "dynamic_columns": 1,
            "category": "Layout",
            "Blocks": [
                {
                    "block_key": block_key,
                    "row": 0,
                    "col": 0,
                    "node_id": PLACEHOLDER_NODE_ID,
                    "output_file_name": OUTPUT_ZIP,
                    "output_file_type": "zip",
                    "Files": [layout_file],
                }
            ],
        },
    }


def write_zip(out_dir: Path = OUT_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    block_key = f"onlb_{secrets.token_hex(8)}.{secrets.randbelow(10**8):08d}"
    script_path = f"{PLACEHOLDER_NODE_ID}/{block_key}/script.py"
    manifest = build_manifest(block_key=block_key, script_path=script_path)
    payload = json.dumps(manifest, sort_keys=True).encode() + SCRIPT_PY.encode()
    digest = hashlib.sha256(payload).hexdigest()[:13]
    zip_path = out_dir / f"NODE_{NODE_NAME}_{digest}.zip"
    stable_path = out_dir / f"NODE_{NODE_NAME}.zip"
    for path in (zip_path, stable_path):
        if path.exists():
            path.unlink()
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(f"{NODE_NAME}.json", json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
            zf.writestr(script_path, SCRIPT_PY)
    return zip_path


def main() -> None:
    path = write_zip(OUT_DIR)
    print(f"wrote {path}")
    print(f"also wrote {OUT_DIR / ('NODE_' + NODE_NAME + '.zip')}")


if __name__ == "__main__":
    main()
