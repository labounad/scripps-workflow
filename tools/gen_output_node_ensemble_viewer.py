#!/usr/bin/env python3
"""Generate GUI Output/Layout node ZIP — standalone ensemble viewer bundle.

The node is intentionally a tiny shim.  It does not contain the viewer assets
or artifact-resolution logic; it imports the implementation from the installed
``scripps-workflow`` package on the HPC.  That makes development fast: update
repo code, ``git pull && pip install -e .`` on the HPC, and rerun.  Re-import
this GUI node only if the GUI-facing inputs/outputs change.

Runtime behavior:
    source pointer/path -> script.py -> ensemble_viewer_bundle.zip

The ``source`` input may be either a ``wf.pointer.v1`` JSON string emitted by a
conformer-producing process node or a concrete XYZ/multi-XYZ path.  The bundle
contains an embedded-payload ``index.html`` for local browser viewing and never
calls the workflow GUI inport resolver.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import zipfile
from pathlib import Path

OUT_DIR = Path("new_nodes_output")
PLACEHOLDER_NODE_ID = 999
NODE_NAME = "ensemble_viewer"
NODE_DESCRIPTION = (
    "Create a downloadable standalone ZIP bundle containing an interactive "
    "3Dmol.js conformer ensemble viewer. The source input may be a "
    "wf.pointer.v1 JSON string or a concrete XYZ/multi-XYZ path."
)
NODE_VERSION = "1.2.0"
OUTPUT_ZIP = "ensemble_viewer_bundle.zip"
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
\"\"\"GUI shim for ensemble_viewer.

All real behavior lives in scripps_workflow.output_viewers.ensemble_bundle.
\"\"\"

from __future__ import annotations

import sys

try:
    from scripps_workflow.output_viewers.ensemble_bundle import main
except Exception as exc:  # pragma: no cover - runtime diagnostics for GUI users
    print(
        "[ensemble_viewer] Could not import scripps_workflow.output_viewers."
        "ensemble_bundle. Make sure scripps-workflow is installed in the HPC "
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
            _input("conformer_index", "text", required=0, tags="optional: all, best, or 1-based index"),
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
