"""Build standalone conformer-ensemble viewer ZIPs from pointers or XYZ files."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

from .artifact_resolver import ArtifactResolutionError, resolve_ensemble_source
from .assets import ENSEMBLE_INDEX_HTML, ENSEMBLE_VIEWER_JS
from .bundle_common import write_bundle
from .cli import parse_viewer_args

DEFAULT_OUTPUT_NAME = "ensemble_viewer_bundle.zip"


def build_bundle(
    *,
    source: str,
    output_name: str = DEFAULT_OUTPUT_NAME,
    smiles: str | None = None,
    title: str | None = None,
    conformer_index: str | None = None,
    cwd: Path | None = None,
) -> Path:
    """Resolve ``source`` and write ``output_name`` under ``cwd``."""

    cwd = Path.cwd() if cwd is None else Path(cwd)
    resolved = resolve_ensemble_source(
        source,
        conformer_index=conformer_index,
        smiles=smiles,
    )
    out = cwd / output_name
    return write_bundle(
        output_path=out,
        viewer_kind="ensemble_viewer",
        index_html=ENSEMBLE_INDEX_HTML,
        viewer_js=ENSEMBLE_VIEWER_JS,
        xyz_text=resolved.xyz_text,
        xyz_file_name="ensemble.xyz",
        title=title or "Conformer Ensemble",
        source_path=resolved.source_path,
        source_label=resolved.label,
        smiles=smiles or resolved.smiles,
        manifest_path=resolved.manifest_path,
        selected_index=resolved.selected_index,
        n_frames=resolved.n_frames,
        metadata=resolved.metadata,
    )


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        args = parse_viewer_args(argv, default_output_name=DEFAULT_OUTPUT_NAME)
        out = build_bundle(
            source=args.source,
            output_name=args.output_name or DEFAULT_OUTPUT_NAME,
            smiles=args.smiles,
            title=args.title,
            conformer_index=args.conformer_index,
        )
    except (ArtifactResolutionError, OSError, ValueError) as e:
        print(f"[ensemble_viewer] ERROR: {e}", file=sys.stderr)
        return 2
    print(str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
