#!/usr/bin/env python3
"""Generate GUI Output/Layout node ZIP — standalone ensemble viewer bundle."""

from __future__ import annotations

import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from gui_export_config import OUT_DIR, InputSpec
from output_node_bundle import LayoutNodeSpec, write_layout_node_zip

NODE_NAME = "ensemble_viewer"
OUTPUT_ZIP = "ensemble_viewer_bundle.zip"

SPEC = LayoutNodeSpec(
    node_name=NODE_NAME,
    description=(
        "Create a downloadable standalone ZIP bundle containing an interactive "
        "3Dmol.js conformer ensemble viewer. The source input may be a "
        "wf.pointer.v1 JSON string or a concrete XYZ/multi-XYZ path."
    ),
    entrypoint="scripps_workflow.output_viewers.ensemble_bundle:main",
    output_zip=OUTPUT_ZIP,
    inputs=(
        InputSpec("source", required=1, tags="wf.pointer.v1 or xyz path"),
        InputSpec("smiles"),
        InputSpec("title"),
        InputSpec("conformer_index", tags="optional: all, best, or 1-based index"),
        InputSpec("output_name", tags="optional zip filename"),
    ),
)


def write_zip(out_dir: Path = OUT_DIR) -> Path:
    return write_layout_node_zip(SPEC, out_dir)


def main() -> None:
    path = write_zip(OUT_DIR)
    print(f"wrote {path}")
    print(f"also wrote {OUT_DIR / ('NODE_' + NODE_NAME + '.zip')}")


if __name__ == "__main__":
    main()
