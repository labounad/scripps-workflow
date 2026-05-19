"""Helpers for generating script-backed Output/Layout node bundles.

The ensemble and geometry viewer nodes have the same GUI shell contract: a tiny
``script.py`` is packaged into an Output/Layout node ZIP and delegates all real
behavior to the installed ``scripps_workflow`` package.  This module centralizes
that fragile shell/bootstrap logic so fixes happen once.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import zipfile
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent

from gui_export_config import (
    AUTHOR,
    DEFAULT_REPO_SRC_CANDIDATES,
    DEFAULT_WORKFLOW_PYTHON,
    NODE_VERSION,
    OUT_DIR,
    PLACEHOLDER_NODE_ID,
    WORKFLOW_HOST,
    InputSpec,
)


@dataclass(frozen=True)
class LayoutNodeSpec:
    """Declarative description of a script-backed GUI Output/Layout node."""

    node_name: str
    description: str
    entrypoint: str
    output_zip: str
    inputs: tuple[InputSpec, ...]
    version: str = NODE_VERSION
    placeholder_node_id: int = PLACEHOLDER_NODE_ID


def _input_dict(spec: InputSpec, *, node_id: int) -> dict:
    return {
        "node_input_id": secrets.randbelow(900000) + 1000,
        "node_id": node_id,
        "name": spec.name,
        "type": spec.type,
        "tags": spec.tags,
        "required": spec.required,
        "new_id": secrets.randbelow(900000) + 1000,
    }


def render_script_py(spec: LayoutNodeSpec) -> str:
    """Return the self-contained ``script.py`` shim for ``spec``."""

    module_name, function_name = spec.entrypoint.split(":", 1)
    repo_candidates = repr(tuple(DEFAULT_REPO_SRC_CANDIDATES))
    return dedent(
        f'''\
        #!/usr/bin/env python3
        """GUI shim for {spec.node_name}.

        All real behavior lives in ``{module_name}``.

        The workflow GUI runs Output/Layout ``script.py`` files with plain
        ``python3``. On the HPC, that interpreter can be a broken system Python
        or a Python without the editable ``scripps-workflow`` package installed.
        This shim therefore does the absolute minimum before re-execing into the
        workflow environment: it imports only ``os`` and ``sys`` and avoids
        ``pathlib``/``re``/other stdlib modules that can fail under mismatched
        Python installs.

        Set ``SCRIPPS_WORKFLOW_NO_REEXEC=1`` to opt out, or
        ``SCRIPPS_WORKFLOW_PYTHON=/path/to/python`` to choose the interpreter.
        """

        import os
        import sys


        DEFAULT_ENV_PY = {DEFAULT_WORKFLOW_PYTHON!r}
        DEFAULT_REPO_SRC_CANDIDATES = {repo_candidates}
        ENTRYPOINT_MODULE = {module_name!r}
        ENTRYPOINT_FUNCTION = {function_name!r}
        NODE_NAME = {spec.node_name!r}


        def _maybe_reexec_into_workflow_python():
            if os.environ.get("SCRIPPS_WORKFLOW_NO_REEXEC") == "1":
                return

            env_py = (
                os.environ.get("SCRIPPS_WORKFLOW_PYTHON")
                or os.environ.get("ENV_PY")
                or DEFAULT_ENV_PY
            )
            env_py = os.path.expanduser(env_py)

            if not os.path.isfile(env_py):
                return

            try:
                same = os.path.realpath(env_py) == os.path.realpath(sys.executable)
            except Exception:
                same = env_py == sys.executable
            if same:
                return

            # The workflow GUI's plain `python3` can leave PYTHONHOME/PYTHONPATH
            # pointing at a system Python. Scrub them before exec and use `-E`
            # so the target interpreter ignores any remaining PYTHON* settings.
            env = os.environ.copy()
            env["SCRIPPS_WORKFLOW_NO_REEXEC"] = "1"
            for key in (
                "PYTHONHOME",
                "PYTHONPATH",
                "PYTHONSTARTUP",
                "PYTHONUSERBASE",
                "PYTHONNOUSERSITE",
            ):
                env.pop(key, None)
            os.execvpe(env_py, [env_py, "-E", __file__, *sys.argv[1:]], env)


        _maybe_reexec_into_workflow_python()


        # Make the output-node deployment directory visible to the package code.
        # The GUI-generated shell uploads a fixed root-level zip, but users also
        # expect artifacts to appear under outputs/<protocol_id>/<block_key>/ in
        # the experiment file panel. The package mirrors the generated zip there.
        os.environ.setdefault(
            "SCRIPPS_VIEWER_OUTPUT_DIR",
            os.path.dirname(os.path.abspath(__file__)),
        )


        # From here on we are either in the workflow Python or explicitly opted out.
        from pathlib import Path


        def _candidate_repo_src_paths():
            candidates = []

            for key in ("SCRIPPS_WORKFLOW_ROOT", "SCRIPPS_WORKFLOW_REPO", "WF_REPO_ROOT"):
                value = os.environ.get(key)
                if value:
                    base = Path(value).expanduser()
                    candidates.extend([base / "src", base])

            cwd = Path.cwd()
            working_directory = Path(os.environ.get("working_directory", str(cwd))).expanduser()
            script_path = Path(__file__).resolve()

            for base in (cwd, working_directory, script_path):
                for parent in [base, *base.parents]:
                    candidates.extend([
                        parent / "src",
                        parent / "scripps-workflow" / "src",
                        parent / "scripps_workflow" / "src",
                    ])

            candidates.extend(Path(p) for p in DEFAULT_REPO_SRC_CANDIDATES)
            candidates.append(Path.home() / "scripps-workflow" / "src")

            seen = set()
            out = []
            for path in candidates:
                try:
                    resolved = path.expanduser().resolve()
                except OSError:
                    resolved = path.expanduser()
                key = str(resolved)
                if key in seen:
                    continue
                seen.add(key)
                if (resolved / "scripps_workflow").is_dir():
                    out.append(resolved)
            return out


        def _add_source_tree_to_path():
            for src in reversed(_candidate_repo_src_paths()):
                s = str(src)
                if s not in sys.path:
                    sys.path.insert(0, s)


        def _diagnostic_import_error(exc):
            src_candidates = _candidate_repo_src_paths()
            print(
                f"[{{NODE_NAME}}] Could not import {{ENTRYPOINT_MODULE}}.\\n"
                f"  Original error: {{exc!r}}\\n"
                f"  sys.executable: {{sys.executable}}\\n"
                f"  cwd: {{Path.cwd()}}\\n"
                f"  script: {{Path(__file__).resolve()}}\\n"
                f"  ENV_PY: {{os.environ.get('ENV_PY', '')}}\\n"
                f"  SCRIPPS_WORKFLOW_PYTHON: {{os.environ.get('SCRIPPS_WORKFLOW_PYTHON', '')}}\\n"
                f"  SCRIPPS_WORKFLOW_ROOT: {{os.environ.get('SCRIPPS_WORKFLOW_ROOT', '')}}\\n"
                f"  detected repo src candidates: {{[str(p) for p in src_candidates]}}\\n"
                "  Fix: on the HPC, run `git pull` in the scripps-workflow repo and "
                "`pip install -e .` in the workflow environment, or set "
                "SCRIPPS_WORKFLOW_ROOT=/path/to/scripps-workflow.",
                file=sys.stderr,
            )


        _add_source_tree_to_path()

        try:
            _module = __import__(ENTRYPOINT_MODULE, fromlist=[ENTRYPOINT_FUNCTION])
            main = getattr(_module, ENTRYPOINT_FUNCTION)
        except Exception as exc:  # pragma: no cover - runtime diagnostics for GUI users
            _diagnostic_import_error(exc)
            raise


        if __name__ == "__main__":
            raise SystemExit(main())
        '''
    )


def build_manifest(spec: LayoutNodeSpec, *, block_key: str, script_path: str) -> dict:
    file_id = secrets.randbelow(900000) + 1000
    file_info = {
        "node_files_info_id": file_id,
        "node_id": spec.placeholder_node_id,
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
        "output_file_name": spec.output_zip,
        "output_file_type": "zip",
        "screeds": 1,
        "cols": 1,
    }
    return {
        "node_id": spec.placeholder_node_id,
        "name": spec.node_name,
        "node_type": "Output",
        "description": spec.description,
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
        "inputs": [_input_dict(input_spec, node_id=spec.placeholder_node_id) for input_spec in spec.inputs],
        "outputs": [],
        "params": [],
        "version": spec.version,
        "host": WORKFLOW_HOST,
        "layout": {
            "node_id": spec.placeholder_node_id,
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
                    "node_id": spec.placeholder_node_id,
                    "output_file_name": spec.output_zip,
                    "output_file_type": "zip",
                    "Files": [layout_file],
                }
            ],
        },
    }


def write_layout_node_zip(spec: LayoutNodeSpec, out_dir: Path = OUT_DIR) -> Path:
    """Write hash-named and stable GUI node ZIPs for ``spec``."""

    out_dir.mkdir(parents=True, exist_ok=True)
    block_key = f"onlb_{secrets.token_hex(8)}.{secrets.randbelow(10**8):08d}"
    script_path = f"{spec.placeholder_node_id}/{block_key}/script.py"
    script_py = render_script_py(spec)
    manifest = build_manifest(spec, block_key=block_key, script_path=script_path)
    payload = json.dumps(manifest, sort_keys=True).encode() + script_py.encode()
    digest = hashlib.sha256(payload).hexdigest()[:13]
    zip_path = out_dir / f"NODE_{spec.node_name}_{digest}.zip"
    stable_path = out_dir / f"NODE_{spec.node_name}.zip"
    for path in (zip_path, stable_path):
        if path.exists():
            path.unlink()
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(
                f"{spec.node_name}.json",
                json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            )
            zf.writestr(script_path, script_py)
    return zip_path
