#!/usr/bin/env python3
"""Export node bundles for the workflow.scripps.edu GUI.

Produces, for each node in :data:`NODES`, a directory tree::

    <out_dir>/<name>/
        <name>.json          # GUI metadata (top-level descriptor)
        0/
            script.sh        # bash shim that exec's the env-py against script.py
            script.py        # python shim that calls the package entry point

and (with ``--zip``) a matching ``NODE_<name>_<8hex>.zip`` next to it,
laid out exactly like the example bundle the GUI exports — so the GUI's
import path round-trips it cleanly.

Usage::

    python tools/export_nodes.py                     # all nodes -> dist/gui_nodes/
    python tools/export_nodes.py --zip               # also write zip bundles
    python tools/export_nodes.py wf-xtb wf-crest     # subset
    python tools/export_nodes.py --out /tmp/foo      # custom out dir

Stdlib-only by design (matches the project's runtime philosophy). The
generated ``script.py`` shims do ``from scripps_workflow.nodes.<mod>
import main; raise SystemExit(main())`` — i.e. the GUI side just needs
``pip install scripps-workflow`` into the env that ``ENV_PY`` points to.

Bumping a node's GUI surface (renaming a config key, adding a new one,
flipping a default) means editing :data:`NODES` here and re-running the
exporter. The actual node implementation lives in
``scripps_workflow.nodes.<module>`` and is the source of truth for
behavior; the registry below is the source of truth for what the GUI
exposes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
#
# One entry per ``wf-*`` console script. Inputs are the names that get wired
# into the GUI's input list, in display order. Convention:
#
#   * Non-source nodes start with ``"pointer JSON"`` — the engine wires the
#     upstream pointer into argv[1] and the rest as ``key=value`` argv tokens.
#   * Source nodes (``smiles_to_3d``, ``tag_input``) omit ``pointer JSON``
#     and start straight with their config keys.
#
# Outputs are the GUI-side output sockets. Every node here emits a single
# ``wf.pointer.v1`` JSON line on stdout, so one output socket is correct.
#
# ``module`` is the dotted path under ``scripps_workflow.nodes``.

DEFAULT_ENV_PY = "/gpfs/group/shenvi/envs/workflow/bin/python"
DEFAULT_HOST = "workflow.scripps.edu"
DEFAULT_VERSION = "1.1.0"
DEFAULT_CATEGORY = "Script"
# Engine convention: every Process node's stdout socket is named "stdout"
# in both the per-node bundle and the workflow connections. The actual
# *content* type flowing on that socket is communicated via the `tags`
# string — typically "pointer JSON" for nodes that emit a wf.pointer.v1
# manifest pointer, and "" for tag nodes that emit a "key=value" token.
DEFAULT_OUTPUT_NAME = "stdout"
DEFAULT_OUTPUT_TAGS = "pointer JSON"


@dataclass
class NodeSpec:
    name: str
    module: str
    description: str = ""
    is_source: bool = False
    inputs: list[str] = field(default_factory=list)  # excludes "pointer JSON"
    outputs: list[str] = field(default_factory=lambda: [DEFAULT_OUTPUT_NAME])
    output_tags: str = DEFAULT_OUTPUT_TAGS
    category: str = DEFAULT_CATEGORY


NODES: dict[str, NodeSpec] = {
    spec.name: spec
    for spec in [
        # --- source nodes (no upstream pointer) ---
        NodeSpec(
            name="wf-embed",
            module="smiles_to_3d",
            description="SMILES -> 3D coordinates (RDKit ETKDG + optional MMFF).",
            is_source=True,
            inputs=[
                "smiles", "name", "molecule",
                "seed", "max_embed_attempts",
                "mmff", "opt", "max_opt_iters",
            ],
        ),
        NodeSpec(
            name="wf-tag-input",
            module="tag_input",
            description=(
                "Engine wiring shim: emit a literal 'key=value' string on stdout. "
                "Unlike every other node here, tag nodes intentionally break the "
                "wf.pointer.v1 contract — downstream consumers parse the token "
                "as a config argv token, not a manifest pointer."
            ),
            is_source=True,
            inputs=["key", "value"],
            output_tags="",
        ),
        # --- conformer / QC pipeline ---
        NodeSpec(
            name="wf-xtb",
            module="xtb_calc",
            description="xTB single-point / optimize / gradient / hessian.",
            inputs=[
                "theory", "calculations", "optimize", "opt_level",
                "charge", "unpaired_electrons", "solvent",
                "threads", "write_json",
            ],
        ),
        NodeSpec(
            name="wf-crest",
            module="crest",
            description="CREST conformer search.",
            inputs=[
                "mode", "theory", "ewin_kcal", "max_conformers",
                "charge", "unpaired_electrons", "solvent", "threads",
            ],
        ),
        NodeSpec(
            name="wf-orca-goat",
            module="orca_goat",
            description="ORCA 6.0+ GOAT global conformer search.",
            inputs=[
                "mode", "theory", "ewin_kcal", "max_conformers",
                "charge", "unpaired_electrons", "solvent",
                "threads", "maxcore_mb",
            ],
        ),
        NodeSpec(
            name="wf-prism",
            module="prism_screen",
            description="Conformer ensemble pruning via prism-pruner (RMSD / MoI / rotation-corrected RMSD).",
            inputs=[
                "rmsd_pruning", "moi_pruning", "rot_corr_rmsd_pruning",
                "use_energies", "ewin_kcal", "max_dE_kcal",
                "min_conformers", "keep_rejected", "timeout_s",
            ],
        ),
        NodeSpec(
            name="wf-marc",
            module="marc_screen",
            description="Conformer ensemble clustering via navicat-marc (sibling of wf-prism).",
            inputs=[
                "metric", "clustering", "n_clusters",
                "use_energies", "ewin_kcal",
                "min_conformers", "keep_rejected", "timeout_s",
            ],
        ),
        # --- ORCA SLURM array nodes ---
        NodeSpec(
            name="wf-orca-dft-array",
            module="orca_dft_array",
            description="SLURM-array DFT geometry optimization (one ORCA call per conformer).",
            inputs=[
                "max_concurrency", "job_name",
                "charge", "unpaired_electrons", "multiplicity",
                "solvent", "smd_solvent", "keywords",
                "maxcore", "nprocs", "time_limit", "partition",
                "orca_module", "submit", "monitor",
                "monitor_interval_s", "monitor_timeout_min",
                "silence_openib",
            ],
        ),
        NodeSpec(
            name="wf-orca-thermo-array",
            module="orca_thermo_array",
            description="SLURM-array composite freq + high-level SP (+ optional NMR shielding/coupling jobs) per conformer.",
            inputs=[
                "max_concurrency", "job_name",
                "charge", "unpaired_electrons", "multiplicity",
                "solvent", "smd_solvent",
                "keywords", "singlepoint_keywords",
                "maxcore", "nprocs", "time_limit", "partition",
                "orca_module", "submit", "monitor",
                "monitor_interval_s", "monitor_timeout_min",
                "silence_openib",
                # NMR knobs
                "run_shielding_h", "shielding_method_h", "shielding_basis_h",
                "run_shielding_c", "shielding_method_c", "shielding_basis_c",
                "run_couplings", "coupling_method", "coupling_basis",
                "coupling_pairs", "coupling_thresh_angstrom",
                "nmr_aux_keywords",
            ],
        ),
        # --- aggregators ---
        NodeSpec(
            name="wf-thermo-aggregate",
            module="thermo_aggregate",
            description="Boltzmann weights + composite Gibbs over a conformer ensemble.",
            inputs=[
                "temperature_k", "standard_state",
                "n_tasks_override", "output_csv",
            ],
        ),
        NodeSpec(
            name="wf-nmr-aggregate",
            module="nmr_aggregate",
            description="Boltzmann-averaged DFT NMR shifts + couplings with cheshire / Bally-Rablen calibration.",
            inputs=[
                "solvent",
                "shielding_method_h", "shielding_basis_h",
                "shielding_method_c", "shielding_basis_c",
                "coupling_method", "coupling_basis",
                "skip_couplings",
                "output_shifts_csv", "output_couplings_csv",
            ],
        ),
    ]
}


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------

SCRIPT_SH_TEMPLATE = """\
#!/bin/bash
set -euo pipefail

# Use your group env python by default; override via ENV_PY if needed.
ENV_PY="${{ENV_PY:-{env_py}}}"

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"

# Prevent Franken-Python via inherited env vars
unset PYTHONHOME PYTHONPATH PYTHONSTARTUP PYTHONNOUSERSITE

exec "$ENV_PY" -E "$SCRIPT_DIR/script.py" "$@"
"""

SCRIPT_PY_TEMPLATE = '''\
#!{env_py}
"""GUI shim for ``{wf_name}`` ({module}).

The actual node implementation lives in
``scripps_workflow.nodes.{module}``. This file just forwards argv.
Generated by ``tools/export_nodes.py`` — do not edit by hand.
"""
from scripps_workflow.nodes.{module} import main


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _make_input_record(name: str, iid: int, node_id: int, *, type_: str = "text") -> dict:
    return {
        "node_input_id": iid,
        "node_id": node_id,
        "name": name,
        "type": type_,
        "tags": "",
        "required": 0,
        "new_id": iid,
    }


def _make_output_record(
    name: str, oid: int, node_id: int, *, type_: str = "text", tags: str = "",
) -> dict:
    return {
        "node_output_id": oid,
        "node_id": node_id,
        "name": name,
        "type": type_,
        "tags": tags,
        "new_id": oid,
    }


def _make_files_info(node_id: int) -> list[dict]:
    return [
        {
            "node_files_info_id": 1,
            "node_id": node_id,
            "file_name": "script.py",
            "file_format": "python",
            "upload_method": "file",
            "path": f"{node_id}/script.py",
        },
        {
            "node_files_info_id": 2,
            "node_id": node_id,
            "file_name": "script.sh",
            "file_format": "bash",
            "upload_method": "code",
            "path": f"{node_id}/script.sh",
        },
    ]


def derive_node_id(name: str) -> int:
    """Deterministic synthetic node_id derived from a node name.

    Used so per-node bundles AND workflow content/connection records can
    agree on the same ``node_id`` value without any prior engine
    assignment. The engine's importer is expected to remap on the way
    in, but the values must be internally consistent within a single
    workflow zip for the connection graph to resolve.
    """
    return int(hashlib.sha1(name.encode()).hexdigest()[:7], 16)


def derive_input_id(node_name: str, input_name: str) -> int:
    h = hashlib.sha1(f"{node_name}:in:{input_name}".encode()).hexdigest()
    return int(h[:7], 16)


def derive_output_id(node_name: str, output_name: str) -> int:
    h = hashlib.sha1(f"{node_name}:out:{output_name}".encode()).hexdigest()
    return int(h[:7], 16)


def build_metadata(
    spec: NodeSpec,
    *,
    env_py: str,
    host: str,
    version: str,
    node_id: int | None = None,
    input_types: dict[str, str] | None = None,
) -> dict:
    """Produce the top-level ``<name>.json`` GUI descriptor for one node.

    :param input_types: Optional override for the GUI ``type`` field of
        named inputs (default ``"text"``). Used by tag-instance bundles
        so a numeric upstream widget surfaces as ``value: number``.
    """
    # node_id derived from the name keeps it stable across exports so a
    # workflow bundle's content/connection records can reference the
    # same value the per-node bundle declares.
    if node_id is None:
        node_id = derive_node_id(spec.name)

    inputs = list(spec.inputs)
    if not spec.is_source:
        inputs = ["pointer JSON", *inputs]

    input_types = input_types or {}

    return {
        "node_id": node_id,
        "name": spec.name,
        "node_type": "Process",
        "description": spec.description,
        "category": spec.category,
        "domain": "private",
        "author": None,  # engine attaches the importing user
        "node_value": "script.sh",
        "directives": "None",
        "is_mpi": 0,
        "processing_type": "",
        "custom_value": 0,
        "file_tracking": 0,
        "limit": "",
        "files_info": _make_files_info(node_id),
        "inputs": [
            _make_input_record(
                n, derive_input_id(spec.name, n), node_id,
                type_=input_types.get(n, "text"),
            )
            for n in inputs
        ],
        "outputs": [
            _make_output_record(
                n, derive_output_id(spec.name, n), node_id,
                tags=spec.output_tags,
            )
            for n in spec.outputs
        ],
        "params": [],
        "version": version,
        "host": host,
    }


def render_script_sh(env_py: str) -> str:
    return SCRIPT_SH_TEMPLATE.format(env_py=env_py)


def render_script_py(spec: NodeSpec, env_py: str) -> str:
    return SCRIPT_PY_TEMPLATE.format(
        env_py=env_py, wf_name=spec.name, module=spec.module,
    )


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def write_node_bundle(
    spec: NodeSpec,
    *,
    out_dir: Path,
    env_py: str,
    host: str,
    version: str,
    make_zip: bool,
) -> tuple[Path, Path | None]:
    bundle_dir = out_dir / spec.name
    sub_dir = bundle_dir / "0"  # node_id placeholder; matches files_info paths
    sub_dir.mkdir(parents=True, exist_ok=True)

    metadata = build_metadata(spec, env_py=env_py, host=host, version=version)
    json_path = bundle_dir / f"{spec.module}.json"
    json_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    sh_path = sub_dir / "script.sh"
    sh_path.write_text(render_script_sh(env_py), encoding="utf-8")
    sh_path.chmod(0o755)

    py_path = sub_dir / "script.py"
    py_path.write_text(render_script_py(spec, env_py), encoding="utf-8")
    py_path.chmod(0o755)

    zip_path: Path | None = None
    if make_zip:
        suffix = secrets.token_hex(7)  # 14 chars (example used 13 — close enough)
        zip_path = out_dir / f"NODE_{spec.module}_{suffix}.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(json_path, arcname=json_path.name)
            zf.write(sh_path, arcname=f"0/{sh_path.name}")
            zf.write(py_path, arcname=f"0/{py_path.name}")

    return bundle_dir, zip_path


def export(
    names: Iterable[str] | None,
    *,
    out_dir: Path,
    env_py: str,
    host: str,
    version: str,
    make_zip: bool,
) -> list[tuple[NodeSpec, Path, Path | None]]:
    if names:
        unknown = [n for n in names if n not in NODES]
        if unknown:
            raise SystemExit(
                f"Unknown node(s): {', '.join(unknown)}\n"
                f"Available: {', '.join(sorted(NODES))}"
            )
        specs = [NODES[n] for n in names]
    else:
        specs = list(NODES.values())

    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[tuple[NodeSpec, Path, Path | None]] = []
    for spec in specs:
        bundle_dir, zip_path = write_node_bundle(
            spec,
            out_dir=out_dir,
            env_py=env_py,
            host=host,
            version=version,
            make_zip=make_zip,
        )
        written.append((spec, bundle_dir, zip_path))
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Export GUI-importable bundles for scripps-workflow nodes."
    )
    p.add_argument(
        "names", nargs="*",
        help="Specific wf-* names to export (default: all).",
    )
    p.add_argument(
        "--out", default="dist/gui_nodes", type=Path,
        help="Output directory (default: %(default)s).",
    )
    p.add_argument(
        "--env-py", default=DEFAULT_ENV_PY,
        help="ENV_PY path baked into the script.sh shim (default: %(default)s).",
    )
    p.add_argument(
        "--host", default=DEFAULT_HOST,
        help="GUI host string in metadata (default: %(default)s).",
    )
    p.add_argument(
        "--version", default=DEFAULT_VERSION,
        help="Metadata version string (default: %(default)s).",
    )
    p.add_argument(
        "--zip", action="store_true",
        help="Also write NODE_<name>_<hex>.zip bundles next to each directory.",
    )
    p.add_argument(
        "--list", action="store_true",
        help="List available nodes and exit.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    if args.list:
        for name, spec in sorted(NODES.items()):
            print(f"{name:24s} -> scripps_workflow.nodes.{spec.module}")
        return 0

    written = export(
        args.names or None,
        out_dir=args.out,
        env_py=args.env_py,
        host=args.host,
        version=args.version,
        make_zip=args.zip,
    )

    for spec, bundle_dir, zip_path in written:
        suffix = f"  zip={zip_path}" if zip_path else ""
        print(f"[ok] {spec.name:22s} -> {bundle_dir}{suffix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
