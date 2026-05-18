"""``wf-extract-conformers`` — bridge conformer manifests to GUI viewer files.

This is the opinionated, low-overhead cousin of ``wf-artifact-export``.
It consumes a normal ``wf.pointer.v1`` JSON pointer, follows the upstream
``manifest.json``, and emits a concrete XYZ file path on stdout for GUI
Layout/Output nodes.

Default behavior is deliberately what the ensemble viewer wants:

    upstream conformer node -> wf-extract-conformers -> ensemble_viewer

With no config tags, the node extracts **all conformers** as one multi-frame
XYZ file. It works with upstream nodes that publish either a canonical
``xyz_ensemble`` artifact (CREST, PRISM, ORCA array nodes) or ordered
per-conformer buckets (``accepted``, ``selected``, ``conformers``).

Optional config is intentionally small:

    conformer_index     optional 1-based conformer index. Empty / ``all``
                        keeps all conformers. ``best`` selects the upstream
                        best ``xyz`` artifact when available.
    output_name         optional filename under this call's ``outputs/``.
    copy_mode           copy | hardlink | symlink | none. Only applies when
                        an existing upstream file can be reused. New combined
                        or extracted files are always written under outputs/.

Like ``wf-artifact-export``, this node deliberately does **not** emit a
``wf.pointer.v1`` pointer. Stdout is exactly one concrete file path because
that is what the GUI-generated ``step_updater.py`` expects for Output nodes.
"""

from __future__ import annotations

import ast
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..hashing import sha256_file
from ..parsing import parse_kv_or_json
from ..pointer import PointerError, load_pointer
from ..schema import ArtifactRecord, Manifest, UpstreamRef
from .crest import XyzBlock, split_multixyz, write_xyz_block


CONFORMER_BUCKET_PRIORITY: tuple[str, ...] = (
    "accepted",
    "selected",
    "conformers",
)


class ExtractConformersError(RuntimeError):
    """Raised for user/config/upstream problems that should fail this node."""


@dataclass(frozen=True)
class SourceRecord:
    bucket: str
    ordinal: int
    record_index: int | None
    label: str | None
    path: Path
    item: dict[str, Any]


def _log(msg: str) -> None:
    print(f"[wf-extract-conformers] {msg}", file=sys.stderr, flush=True)


def _environment() -> dict[str, Any]:
    try:
        host = socket.gethostname()
    except Exception:
        host = None
    return {
        "python": sys.version.split()[0],
        "python_exe": sys.executable,
        "platform": platform.platform(),
        "host": host,
    }


def _artifact_items(artifacts: Mapping[str, Any], key: str) -> list[dict[str, Any]]:
    raw = artifacts.get(key)
    if isinstance(raw, list):
        out: list[dict[str, Any]] = []
        for item in raw:
            if isinstance(item, Mapping):
                out.append(dict(item))
            elif isinstance(item, str):
                out.append({"path_abs": item})
        return out
    if isinstance(raw, str):
        return [{"path_abs": raw}]
    return []


def _item_path(item: Mapping[str, Any], *, upstream_manifest: Manifest) -> Path | None:
    raw = item.get("path_abs") or item.get("path") or item.get("path_rel")
    if not isinstance(raw, str) or not raw.strip():
        return None
    p = Path(raw).expanduser()
    if not p.is_absolute():
        base = Path(upstream_manifest.cwd) if upstream_manifest.cwd else Path.cwd()
        p = base / p
    return p.resolve()


def _safe_output_name(default_name: str, requested: Any) -> str:
    if requested is None or str(requested).strip().lower() in {"", "auto", "none", "null"}:
        return Path(default_name).name
    name = Path(str(requested).strip()).name
    if not name or name in {".", ".."}:
        raise ExtractConformersError(f"invalid output_name={requested!r}")
    return name


def _copy_file(source: Path, dest: Path, *, copy_mode: str) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    mode = str(copy_mode or "copy").strip().lower()
    if mode in {"", "auto"}:
        mode = "copy"

    if mode == "none":
        return source.resolve()

    if dest.exists() or dest.is_symlink():
        dest.unlink()

    if mode == "copy":
        shutil.copy2(source, dest)
        return dest.resolve()
    if mode == "hardlink":
        try:
            os.link(source, dest)
        except OSError:
            shutil.copy2(source, dest)
        return dest.resolve()
    if mode == "symlink":
        dest.symlink_to(source)
        return dest.resolve()

    raise ExtractConformersError(
        f"unknown copy_mode={copy_mode!r}; expected copy, hardlink, symlink, or none"
    )


def _protocol_id_from_call_dir(cwd: Path) -> int | None:
    """Return the trailing GUI protocol id from a call directory name.

    Example: ``wf_extract_conformers_imported_105443`` -> ``105443``.
    """
    m = re.search(r"_(\d+)$", cwd.name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _experiment_root_from_call_dir(cwd: Path) -> Path | None:
    """Infer the experiment/run root from ``.../calls/<this-call>``."""
    if cwd.parent.name == "calls":
        return cwd.parent.parent.resolve()
    return None


def _step_protocol_map(script_text: str) -> dict[int, int]:
    """Parse the GUI-generated ``-blanks -json`` step table.

    The output-node shell blocks do not set ``protocol_id`` themselves, so
    the only reliable way for a process node to discover the downstream
    Output/Layout protocol id is the step table emitted at the top of the
    generated script.
    """
    m = re.search(r"-blanks\s+-json\s+\"(.+?)\"", script_text, flags=re.DOTALL)
    if not m:
        return {}
    raw = m.group(1)
    try:
        rows = ast.literal_eval(raw)
    except Exception:
        return {}

    out: dict[int, int] = {}
    if not isinstance(rows, list):
        return out
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            out[int(row["step"])] = int(row["protocol_id"])
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _output_steps_consuming_file(script_text: str, *, source_protocol_id: int) -> list[int]:
    """Return GUI step numbers whose output block consumes ``file_<pid>``."""
    wanted = f"result_f=$file_{source_protocol_id}"
    steps: list[int] = []
    # Keep this intentionally simple/robust against cosmetic script changes:
    # split on START markers and search each block for the exact assignment.
    pattern = re.compile(
        r"# START: Step\s+(\d+)\s+-+>\n(?P<body>.*?)(?=# START: Step|\Z)",
        flags=re.DOTALL,
    )
    for m in pattern.finditer(script_text):
        body = m.group("body")
        if wanted in body and "-download" in body:
            try:
                steps.append(int(m.group(1)))
            except ValueError:
                pass
    return steps


@dataclass(frozen=True)
class OutputBinding:
    """A GUI Output/Layout step that consumes this node's file output."""
    step: int
    protocol_id: int
    protocol_name: str
    podid: int


def _step_metadata(script_text: str) -> dict[int, tuple[int, str]]:
    """Return ``step -> (protocol_id, details/name)`` from the generated script.

    The GUI-generated shell block for Output/Layout nodes currently does not
    reset ``protocol_id`` / ``protocol_name`` before calling ``step_updater.py
    -download``. The step table at the top of the script is therefore the only
    reliable place to recover the *actual* output node protocol id/name.
    """
    m = re.search(r"-blanks\s+-json\s+\"(.+?)\"", script_text, flags=re.DOTALL)
    if not m:
        return {}
    try:
        rows = ast.literal_eval(m.group(1))
    except Exception:
        return {}

    out: dict[int, tuple[int, str]] = {}
    if not isinstance(rows, list):
        return out
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            step = int(row["step"])
            pid = int(row["protocol_id"])
            name = str(row.get("details") or "")
        except (KeyError, TypeError, ValueError):
            continue
        out[step] = (pid, name)
    return out


def _output_bindings_consuming_file(
    script_text: str,
    *,
    source_protocol_id: int,
) -> list[OutputBinding]:
    """Return output-node bindings that consume ``file_<source_protocol_id>``.

    A generated Output/Layout block currently looks like::

        # START: Step 20 ...
        result_f=$file_105656
        python3 ... step_updater.py ... -download -podid 90131 ...

    but it omits ``protocol_id="105658"`` for the output node itself. This
    parser pairs the block with the top-level step table so we can self-register
    the concrete file path against the correct output node before the browser
    tries to resolve its inport.
    """
    step_meta = _step_metadata(script_text)
    wanted = f"result_f=$file_{source_protocol_id}"
    bindings: list[OutputBinding] = []
    pattern = re.compile(
        r"# START: Step\s+(\d+)\s+-+>\n(?P<body>.*?)(?=# START: Step|\Z)",
        flags=re.DOTALL,
    )
    for m in pattern.finditer(script_text):
        body = m.group("body")
        if wanted not in body or "-download" not in body:
            continue
        try:
            step = int(m.group(1))
        except ValueError:
            continue
        if step not in step_meta:
            continue
        podid_m = re.search(r"-podid\s+(\d+)", body)
        if not podid_m:
            _log(f"output registration skipped for step {step}: no -podid found")
            continue
        pid, name = step_meta[step]
        try:
            podid = int(podid_m.group(1))
        except ValueError:
            continue
        bindings.append(OutputBinding(step=step, protocol_id=pid, protocol_name=name, podid=podid))
    return bindings


def _register_file_for_output_viewers(
    *,
    exported_path: Path,
) -> list[dict[str, Any]]:
    """Self-register ``exported_path`` for linked Output/Layout viewers.

    This is the critical bridge for the Scripps Workflow GUI. ``step_updater.py``
    is generated by the platform, and for Output/Layout nodes its ``-download``
    call can inherit a stale ``protocol_id``/``protocol_name`` from a previous
    process step. When that happens, the browser-side
    ``get_inputs_for_output_node`` resolver has no file associated with the
    actual viewer protocol id and the viewer spins forever or reports
    ``Failed to fetch``.

    Because this process node can infer the downstream output bindings from
    the generated shell script, it proactively calls the same ``step_updater.py
    -download`` service with the *correct* output protocol id/name/podid.
    The later stale generated call is harmless; this correct registration is
    what the viewer should resolve.
    """
    if not exported_path or not exported_path.is_file():
        return []

    cwd = Path.cwd().resolve()
    run_root = _experiment_root_from_call_dir(cwd)
    source_protocol_id = _protocol_id_from_call_dir(cwd)
    if run_root is None or source_protocol_id is None:
        _log("viewer registration skipped: could not infer run root/protocol id")
        return []

    experiment_id = os.environ.get("experiment_id") or run_root.name
    step_updater = run_root / "step_updater.py"
    script_path = run_root / "script"
    if not step_updater.is_file():
        _log("viewer registration skipped: step_updater.py not found")
        return []
    if not script_path.is_file():
        _log("viewer registration skipped: generated workflow script not found")
        return []

    try:
        script_text = script_path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        _log(f"viewer registration skipped: could not read script: {e}")
        return []

    bindings = _output_bindings_consuming_file(
        script_text,
        source_protocol_id=source_protocol_id,
    )
    if not bindings:
        _log("viewer registration: no linked output viewer bindings found")
        return []

    registered: list[dict[str, Any]] = []
    for b in bindings:
        step_updater_python = os.environ.get("WF_STEP_UPDATER_PYTHON", "python3")
        cmd = [
            step_updater_python,
            str(step_updater),
            "-e",
            str(experiment_id),
            "-pid",
            str(b.protocol_id),
            "-pname",
            b.protocol_name,
            "-download",
            "-podid",
            str(b.podid),
            "-body",
            str(exported_path.resolve()),
        ]
        rec: dict[str, Any] = {
            "step": b.step,
            "output_protocol_id": b.protocol_id,
            "output_protocol_name": b.protocol_name,
            "podid": b.podid,
            "file_path": str(exported_path.resolve()),
        }
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(run_root),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            rec["returncode"] = proc.returncode
            if proc.stdout.strip():
                rec["stdout"] = proc.stdout.strip()
            if proc.stderr.strip():
                rec["stderr"] = proc.stderr.strip()
            if proc.returncode == 0:
                _log(
                    "registered viewer download -> "
                    f"pid={b.protocol_id} pname={b.protocol_name} "
                    f"podid={b.podid} file={exported_path}"
                )
            else:
                _log(
                    "viewer registration failed -> "
                    f"pid={b.protocol_id} podid={b.podid} rc={proc.returncode}"
                )
        except Exception as e:
            rec["error"] = str(e)
            _log(f"viewer registration exception: {e}")
        registered.append(rec)

    return registered


def _schedule_delayed_viewer_registrations(
    *,
    exported_path: Path,
) -> list[dict[str, Any]]:
    """Schedule post-output-step registration retries for linked viewers.

    The GUI-generated Output/Layout shell block runs *after* this process node
    and currently calls ``step_updater.py -download`` with stale
    ``protocol_id`` / ``protocol_name`` variables. In practice that later stale
    call can clobber the correct registration made synchronously by this node.

    To make the bridge robust without editing GUI-generated ``step_updater.py``
    or the generated workflow script, write and launch a tiny background shell
    script that repeats the correct registration after the output step has had
    time to run. The retry log is intentionally written under this call's
    ``outputs/`` directory so exported run zips show exactly what happened.
    """
    if not exported_path or not exported_path.is_file():
        return []

    cwd = Path.cwd().resolve()
    run_root = _experiment_root_from_call_dir(cwd)
    source_protocol_id = _protocol_id_from_call_dir(cwd)
    if run_root is None or source_protocol_id is None:
        _log("delayed viewer registration skipped: could not infer run root/protocol id")
        return []

    step_updater = run_root / "step_updater.py"
    script_path = run_root / "script"
    if not step_updater.is_file() or not script_path.is_file():
        _log("delayed viewer registration skipped: step_updater.py or script missing")
        return []

    try:
        script_text = script_path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        _log(f"delayed viewer registration skipped: could not read script: {e}")
        return []

    bindings = _output_bindings_consuming_file(script_text, source_protocol_id=source_protocol_id)
    if not bindings:
        _log("delayed viewer registration: no linked output viewer bindings found")
        return []

    experiment_id = os.environ.get("experiment_id") or run_root.name
    step_updater_python = os.environ.get("WF_STEP_UPDATER_PYTHON", "python3")
    outputs_dir = cwd / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    retry_script = outputs_dir / "viewer_registration_retry.sh"
    retry_log = outputs_dir / "viewer_registration_retry.log"

    lines = [
        "#!/usr/bin/env bash",
        "set +e",
        f"LOG={str(retry_log)!r}",
        "echo \"[viewer-registration-retry] start $(date -Is) host=$(hostname)\" >> \"$LOG\"",
    ]
    # Cumulative delays: 2s, 8s, 20s, 45s, 90s. This covers both immediate
    # output-step clobbering and a user opening/reloading the viewer shortly
    # after the process node finishes.
    for delay in (2, 6, 12, 25, 45):
        lines.append(f"sleep {delay}")
        for b in bindings:
            cmd = [
                step_updater_python,
                str(step_updater),
                "-e", str(experiment_id),
                "-pid", str(b.protocol_id),
                "-pname", b.protocol_name,
                "-download",
                "-podid", str(b.podid),
                "-body", str(exported_path.resolve()),
            ]
            quoted = " ".join(shlex_quote(x) for x in cmd)
            lines.extend([
                f"echo \"[viewer-registration-retry] $(date -Is) registering pid={b.protocol_id} pname={b.protocol_name} podid={b.podid} file={exported_path.resolve()}\" >> \"$LOG\"",
                f"{quoted} >> \"$LOG\" 2>&1",
                "rc=$?",
                "echo \"[viewer-registration-retry] rc=${rc}\" >> \"$LOG\"",
            ])
    lines.append("echo \"[viewer-registration-retry] done $(date -Is)\" >> \"$LOG\"")
    retry_script.write_text("\n".join(lines) + "\n", encoding="utf-8")
    retry_script.chmod(0o755)

    try:
        subprocess.Popen(
            ["bash", str(retry_script)],
            cwd=str(run_root),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        _log(f"scheduled delayed viewer registration retries -> {retry_log}")
    except Exception as e:
        _log(f"failed to schedule delayed viewer registration retries: {e}")

    return [
        {
            "delayed_registration": {
                "script": str(retry_script.resolve()),
                "log": str(retry_log.resolve()),
                "bindings": [
                    {
                        "step": b.step,
                        "output_protocol_id": b.protocol_id,
                        "output_protocol_name": b.protocol_name,
                        "podid": b.podid,
                    }
                    for b in bindings
                ],
            }
        }
    ]


def shlex_quote(s: object) -> str:
    """Small local quote helper to avoid importing shlex in old bundles."""
    text = str(s)
    if re.match(r"^[A-Za-z0-9_@%+=:,./-]+$", text):
        return text
    return "'" + text.replace("'", "'\\''") + "'"


def _embed_viewer_payload(block_dir: Path, *, meta: dict[str, Any], xyz_path: Path) -> Path | None:
    """Embed the viewer payload directly into an output block's index.html.

    The workflow GUI iframe has proven unreliable for fetching either inport
    resolver endpoints or sibling ``data/*.json`` files. The one thing the GUI
    must serve is the output block's ``index.html``. Embedding the XYZ text in
    that HTML gives the viewer a zero-network primary path. The sidecar data
    files are still written as useful artifacts/fallbacks.
    """
    index_path = block_dir / "index.html"
    if not index_path.is_file():
        _log(f"viewer embed skipped: missing {index_path}")
        return None

    try:
        html = index_path.read_text(encoding="utf-8", errors="replace")
        xyz_text = xyz_path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        _log(f"viewer embed skipped: could not read payload/html: {e}")
        return None

    payload = dict(meta)
    payload["inline"] = True
    payload["xyz_text"] = xyz_text
    payload_json = json.dumps(payload, sort_keys=True)
    # Avoid ever closing the script tag from data, even though XYZ should not
    # normally contain this sequence.
    payload_json = payload_json.replace("</", "<\\/")
    tag = (
        '<script id="scripps-viewer-input" type="application/json">\n'
        f'{payload_json}\n'
        '</script>'
    )

    pattern = re.compile(
        r'\n?\s*<script\s+id=["\']scripps-viewer-input["\']\s+type=["\']application/json["\']>.*?</script>\s*',
        flags=re.DOTALL | re.IGNORECASE,
    )
    if pattern.search(html):
        new_html = pattern.sub("\n" + tag + "\n", html, count=1)
    elif "</body>" in html:
        new_html = html.replace("</body>", tag + "\n</body>", 1)
    elif "</html>" in html:
        new_html = html.replace("</html>", tag + "\n</html>", 1)
    else:
        new_html = html + "\n" + tag + "\n"

    index_path.write_text(new_html, encoding="utf-8")
    return index_path.resolve()


def _stage_for_output_viewers(
    *,
    exported_path: Path,
    output_kind: str | None,
    inputs: dict[str, Any],
) -> list[dict[str, Any]]:
    """Copy exported XYZ into linked Output/Layout viewer directories.

    Why this exists:
      The workflow GUI serves Layout nodes in an iframe, but browser-side
      calls to ``get_inputs_for_output_node`` can fail because the iframe has
      no bearer token and cross-origin requests to the legacy opaat endpoint
      are blocked by CORS. Since this process node already has the concrete
      file and runs on the same filesystem as the generated output block,
      staging the data next to the viewer is the most robust contract.

    The generated top-level script contains enough information to pair this
    call with output blocks that consume ``$file_<this_protocol_id>``. For
    each linked viewer, write:

      outputs/<output_protocol_id>/onlb_*/data/viewer_input.json
      outputs/<output_protocol_id>/onlb_*/data/ensemble.xyz or geometry.xyz

    It also embeds the same payload directly into index.html as an
    application/json script tag. The direct embed is the primary path: some
    workflow GUI deployments serve the output node HTML/JS but do not expose
    newly-created sibling data files through the same static route. Embedding
    in the already-served HTML avoids all browser-side inport/API fetches.
    """
    if not exported_path or not exported_path.is_file():
        return []

    cwd = Path.cwd().resolve()
    run_root = _experiment_root_from_call_dir(cwd)
    source_protocol_id = _protocol_id_from_call_dir(cwd)
    if run_root is None or source_protocol_id is None:
        _log("viewer staging skipped: could not infer run root/protocol id")
        return []

    script_path = run_root / "script"
    if not script_path.is_file():
        _log("viewer staging skipped: generated workflow script not found")
        return []

    try:
        script_text = script_path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        _log(f"viewer staging skipped: could not read script: {e}")
        return []

    step_to_protocol = _step_protocol_map(script_text)
    consuming_steps = _output_steps_consuming_file(
        script_text,
        source_protocol_id=source_protocol_id,
    )
    output_protocol_ids = [
        step_to_protocol[s] for s in consuming_steps if s in step_to_protocol
    ]

    if not output_protocol_ids:
        _log("viewer staging: no linked output viewer blocks found")
        return []

    staged: list[dict[str, Any]] = []
    default_file = "ensemble.xyz" if output_kind == "ensemble" else "geometry.xyz"
    for output_protocol_id in output_protocol_ids:
        output_dir = run_root / "outputs" / str(output_protocol_id)
        if not output_dir.is_dir():
            _log(f"viewer staging skipped: missing output dir {output_dir}")
            continue

        block_dirs = [p for p in output_dir.iterdir() if p.is_dir() and p.name.startswith("onlb_")]
        if not block_dirs:
            _log(f"viewer staging skipped: no onlb_* dir under {output_dir}")
            continue

        for block_dir in block_dirs:
            data_dir = block_dir / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            staged_xyz = data_dir / default_file
            shutil.copy2(exported_path, staged_xyz)

            meta = {
                "schema": "scripps.viewer_input.v1",
                "file": staged_xyz.name,
                "source_protocol_id": source_protocol_id,
                "output_protocol_id": output_protocol_id,
                "output_kind": output_kind,
                "source_path_abs": str(exported_path.resolve()),
            }
            # Preserve a SMILES value in the sidecar when the user passes one
            # through config. Most workflows do not, so viewers still degrade
            # gracefully without the 2D inset/SMILES block.
            smiles = inputs.get("smiles") or inputs.get("SMILES")
            if smiles:
                meta["smiles"] = str(smiles)

            meta_path = data_dir / "viewer_input.json"
            meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            embedded_index = _embed_viewer_payload(block_dir, meta=meta, xyz_path=staged_xyz)
            staged.append(
                {
                    "output_protocol_id": output_protocol_id,
                    "block_dir": str(block_dir.resolve()),
                    "xyz_path": str(staged_xyz.resolve()),
                    "manifest_path": str(meta_path.resolve()),
                    "embedded_index": str(embedded_index) if embedded_index else None,
                }
            )
            if embedded_index:
                _log(f"embedded viewer data -> {embedded_index}")
            _log(f"staged viewer data -> {staged_xyz}")

    return staged


def _sort_key(item: dict[str, Any], fallback: int) -> tuple[int, int]:
    idx = item.get("index")
    try:
        return (0, int(idx))
    except (TypeError, ValueError):
        return (1, fallback)


def _ordered_source_records(upstream_manifest: Manifest) -> list[SourceRecord]:
    artifacts = upstream_manifest.artifacts or {}
    for bucket in CONFORMER_BUCKET_PRIORITY:
        items = _artifact_items(artifacts, bucket)
        candidates: list[tuple[tuple[int, int], dict[str, Any], Path]] = []
        for pos, item in enumerate(items, start=1):
            path = _item_path(item, upstream_manifest=upstream_manifest)
            if path is None or not path.is_file():
                continue
            if path.suffix.lower() != ".xyz":
                continue
            candidates.append((_sort_key(item, pos), item, path))

        if not candidates:
            continue

        candidates.sort(key=lambda t: t[0])
        records: list[SourceRecord] = []
        for ordinal, (_, item, path) in enumerate(candidates, start=1):
            rec_index: int | None = None
            try:
                rec_index = int(item["index"])
            except (KeyError, TypeError, ValueError):
                pass
            label = item.get("label") if isinstance(item.get("label"), str) else None
            records.append(
                SourceRecord(
                    bucket=bucket,
                    ordinal=ordinal,
                    record_index=rec_index,
                    label=label,
                    path=path,
                    item=dict(item),
                )
            )
        return records

    return []


def _first_xyz_ensemble(upstream_manifest: Manifest) -> tuple[dict[str, Any], Path] | None:
    artifacts = upstream_manifest.artifacts or {}
    for item in _artifact_items(artifacts, "xyz_ensemble"):
        path = _item_path(item, upstream_manifest=upstream_manifest)
        if path is not None and path.is_file() and path.suffix.lower() == ".xyz":
            return dict(item), path
    return None


def _best_xyz(upstream_manifest: Manifest) -> tuple[dict[str, Any], Path] | None:
    artifacts = upstream_manifest.artifacts or {}
    xyz_items = _artifact_items(artifacts, "xyz")
    if not xyz_items:
        return None

    # Prefer an explicitly-labeled best geometry; otherwise first usable xyz.
    sorted_items = sorted(
        xyz_items,
        key=lambda item: 0 if str(item.get("label") or "").lower() == "best" else 1,
    )
    for item in sorted_items:
        path = _item_path(item, upstream_manifest=upstream_manifest)
        if path is not None and path.is_file() and path.suffix.lower() == ".xyz":
            return dict(item), path
    return None


def _conformer_index_token(raw: Any) -> str:
    if raw is None:
        return "all"
    token = str(raw).strip().lower()
    if token in {"", "auto", "none", "null", "all", "ensemble", "conformers"}:
        return "all"
    return token


def _select_source_record(records: list[SourceRecord], requested: int) -> SourceRecord:
    if requested < 1:
        raise ExtractConformersError("conformer_index must be >= 1, 'all', or 'best'")

    for rec in records:
        if rec.record_index == requested:
            return rec

    if requested <= len(records):
        return records[requested - 1]

    raise ExtractConformersError(
        f"conformer_index {requested} not available; only {len(records)} conformer(s) found"
    )


def _concat_source_records(records: Iterable[SourceRecord], out_path: Path) -> None:
    chunks: list[str] = []
    for rec in records:
        text = rec.path.read_text(encoding="utf-8", errors="replace")
        if not text.endswith("\n"):
            text += "\n"
        chunks.append(text)
    if not chunks:
        raise ExtractConformersError("no conformer xyz records to concatenate")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(chunks), encoding="utf-8")


def _extract_frame_from_ensemble(
    ensemble_path: Path,
    *,
    conformer_index: int,
    out_path: Path,
) -> XyzBlock:
    blocks = split_multixyz(ensemble_path.read_text(encoding="utf-8", errors="replace"))
    if not blocks:
        raise ExtractConformersError(f"no XYZ frames parsed from {ensemble_path}")
    if conformer_index < 1 or conformer_index > len(blocks):
        raise ExtractConformersError(
            f"conformer_index {conformer_index} out of range for "
            f"{ensemble_path.name} with {len(blocks)} frame(s)"
        )
    block = blocks[conformer_index - 1]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_xyz_block(out_path, block)
    return block


def _write_manifest(
    *,
    manifest_path: Path,
    started_at_perf: float,
    ok: bool,
    inputs: dict[str, Any],
    upstream: UpstreamRef,
    exported_path: Path | None,
    output_kind: str | None,
    source_kind: str | None,
    source_path: Path | None,
    source_records: list[SourceRecord],
    staged_viewers: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> None:
    manifest = Manifest.skeleton(
        step="extract_conformers",
        cwd=Path.cwd(),
        upstream=upstream,
    )
    manifest.ok = ok
    manifest.created_at_unix = int(time.time())
    manifest.runtime_seconds = float(time.perf_counter() - started_at_perf)
    manifest.inputs.update(inputs)
    manifest.environment.update(_environment())
    manifest.failures.extend(failures)
    if staged_viewers:
        manifest.inputs["staged_viewers"] = staged_viewers

    if exported_path is not None and exported_path.is_file():
        try:
            digest = sha256_file(exported_path)
        except Exception:
            digest = None
        base_record = ArtifactRecord(
            path_abs=str(exported_path.resolve()),
            label="extracted_conformers" if output_kind == "ensemble" else "extracted_conformer",
            format="xyz",
            sha256=digest,
            extra={
                "stdout_contract": "file_path_for_gui_output_node",
                "output_kind": output_kind,
                "source_kind": source_kind,
                "source_path_abs": str(source_path.resolve()) if source_path else None,
                "source_records": [
                    {
                        "bucket": r.bucket,
                        "ordinal": r.ordinal,
                        "index": r.record_index,
                        "label": r.label,
                        "path_abs": str(r.path.resolve()),
                    }
                    for r in source_records
                ],
            },
        )
        manifest.add_artifact("files", base_record)
        if output_kind == "ensemble":
            manifest.add_artifact(
                "xyz_ensemble",
                ArtifactRecord(
                    path_abs=str(exported_path.resolve()),
                    label="extracted_conformers",
                    format="xyz",
                    sha256=digest,
                ),
            )
        elif output_kind == "single":
            manifest.add_artifact(
                "xyz",
                ArtifactRecord(
                    path_abs=str(exported_path.resolve()),
                    label="extracted_conformer",
                    format="xyz",
                    sha256=digest,
                ),
            )

    manifest.write(manifest_path)


def _run(argv: Iterable[str]) -> int:
    argv_list = list(argv)
    started_at_perf = time.perf_counter()
    cwd = Path.cwd().resolve()
    outputs_dir = cwd / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = outputs_dir / "manifest.json"

    inputs: dict[str, Any] = {"raw_argv": argv_list}
    upstream_ref = UpstreamRef()
    failures: list[dict[str, Any]] = []
    exported_path: Path | None = None
    output_kind: str | None = None
    source_kind: str | None = None
    source_path: Path | None = None
    selected_records: list[SourceRecord] = []
    staged_viewers: list[dict[str, Any]] = []

    try:
        if len(argv_list) < 2:
            raise ExtractConformersError("missing argv[1]: upstream wf.pointer.v1 JSON")

        pointer = load_pointer(argv_list[1])
        upstream_ref = UpstreamRef(
            pointer_schema=pointer.schema,
            ok=pointer.ok,
            manifest_path=pointer.manifest_path,
        )
        if not pointer.ok:
            raise ExtractConformersError("upstream pointer has ok=false")

        upstream_manifest_path = Path(pointer.manifest_path)
        if not upstream_manifest_path.is_file():
            raise ExtractConformersError(f"upstream manifest not found: {upstream_manifest_path}")
        upstream_manifest = Manifest.read(upstream_manifest_path)
        if not upstream_manifest.ok:
            raise ExtractConformersError("upstream manifest has ok=false")

        cfg = parse_kv_or_json(argv_list[2:]) if len(argv_list) > 2 else {}
        inputs.update(cfg)

        token = _conformer_index_token(cfg.get("conformer_index"))
        copy_mode = str(cfg.get("copy_mode") or "copy")

        if token == "all":
            ensemble = _first_xyz_ensemble(upstream_manifest)
            if ensemble is not None:
                item, source_path = ensemble
                source_kind = "xyz_ensemble"
                output_kind = "ensemble"
                default_name = source_path.name or "extracted_conformers.xyz"
                output_name = _safe_output_name(default_name, cfg.get("output_name"))
                exported_path = _copy_file(
                    source_path,
                    outputs_dir / output_name,
                    copy_mode=copy_mode,
                )
                inputs.update(
                    {
                        "resolved_mode": "all",
                        "resolved_source": "xyz_ensemble",
                        "n_extracted": len(
                            split_multixyz(
                                source_path.read_text(encoding="utf-8", errors="replace")
                            )
                        ),
                    }
                )
            else:
                records = _ordered_source_records(upstream_manifest)
                if not records:
                    raise ExtractConformersError(
                        "no conformers found; expected xyz_ensemble or one of "
                        f"{', '.join(CONFORMER_BUCKET_PRIORITY)}"
                    )
                selected_records = records
                source_kind = records[0].bucket
                source_path = records[0].path
                output_kind = "ensemble"
                output_name = _safe_output_name(
                    "extracted_conformers.xyz",
                    cfg.get("output_name"),
                )
                exported_path = (outputs_dir / output_name).resolve()
                _concat_source_records(records, exported_path)
                inputs.update(
                    {
                        "resolved_mode": "all",
                        "resolved_source": source_kind,
                        "n_extracted": len(records),
                    }
                )

        elif token == "best":
            best = _best_xyz(upstream_manifest)
            if best is None:
                records = _ordered_source_records(upstream_manifest)
                if not records:
                    raise ExtractConformersError("no best xyz or conformer records found")
                rec = records[0]
                selected_records = [rec]
                source_path = rec.path
                source_kind = rec.bucket
            else:
                _, source_path = best
                source_kind = "xyz"
            output_kind = "single"
            output_name = _safe_output_name("best.xyz", cfg.get("output_name"))
            exported_path = _copy_file(
                source_path,
                outputs_dir / output_name,
                copy_mode=copy_mode,
            )
            inputs.update(
                {"resolved_mode": "best", "resolved_source": source_kind, "n_extracted": 1}
            )

        else:
            try:
                requested_index = int(token)
            except ValueError as e:
                raise ExtractConformersError(
                    "conformer_index must be an integer, 'all', or 'best'"
                ) from e

            records = _ordered_source_records(upstream_manifest)
            if records:
                rec = _select_source_record(records, requested_index)
                selected_records = [rec]
                source_path = rec.path
                source_kind = rec.bucket
                output_kind = "single"
                idx_for_name = rec.record_index if rec.record_index is not None else rec.ordinal
                output_name = _safe_output_name(
                    f"conformer_{idx_for_name:04d}.xyz",
                    cfg.get("output_name"),
                )
                exported_path = _copy_file(
                    source_path,
                    outputs_dir / output_name,
                    copy_mode=copy_mode,
                )
                inputs.update(
                    {
                        "resolved_mode": "single",
                        "resolved_source": source_kind,
                        "requested_conformer_index": requested_index,
                        "selected_record_index": rec.record_index,
                        "selected_ordinal": rec.ordinal,
                        "n_extracted": 1,
                    }
                )
            else:
                ensemble = _first_xyz_ensemble(upstream_manifest)
                if ensemble is None:
                    raise ExtractConformersError(
                        "no conformers found; expected xyz_ensemble or per-conformer records"
                    )
                _, source_path = ensemble
                source_kind = "xyz_ensemble"
                output_kind = "single"
                output_name = _safe_output_name(
                    f"conformer_{requested_index:04d}.xyz",
                    cfg.get("output_name"),
                )
                exported_path = (outputs_dir / output_name).resolve()
                _extract_frame_from_ensemble(
                    source_path,
                    conformer_index=requested_index,
                    out_path=exported_path,
                )
                inputs.update(
                    {
                        "resolved_mode": "single",
                        "resolved_source": "xyz_ensemble",
                        "requested_conformer_index": requested_index,
                        "n_extracted": 1,
                    }
                )

        _log(f"exported -> {exported_path}")
        staged_viewers = _stage_for_output_viewers(
            exported_path=exported_path,
            output_kind=output_kind,
            inputs=inputs,
        )
        registered_viewers = _register_file_for_output_viewers(
            exported_path=exported_path,
        )
        if registered_viewers:
            staged_viewers.extend(
                {"registration": r} for r in registered_viewers
            )
        delayed_registrations = _schedule_delayed_viewer_registrations(
            exported_path=exported_path,
        )
        if delayed_registrations:
            staged_viewers.extend(delayed_registrations)
        _write_manifest(
            manifest_path=manifest_path,
            started_at_perf=started_at_perf,
            ok=True,
            inputs=inputs,
            upstream=upstream_ref,
            exported_path=exported_path,
            output_kind=output_kind,
            source_kind=source_kind,
            source_path=source_path,
            source_records=selected_records,
            staged_viewers=staged_viewers,
            failures=failures,
        )

        # LOAD-BEARING: stdout is consumed by GUI output-node step_updater.py.
        # It must be exactly one concrete file path, not a wf.pointer.v1 JSON.
        sys.stdout.write(str(exported_path) + "\n")
        sys.stdout.flush()
        return 0

    except (ExtractConformersError, PointerError, Exception) as e:
        failures.append({"error": str(e)})
        _log(f"ERROR: {e}")
        try:
            _write_manifest(
                manifest_path=manifest_path,
                started_at_perf=started_at_perf,
                ok=False,
                inputs=inputs,
                upstream=upstream_ref,
                exported_path=exported_path,
                output_kind=output_kind,
                source_kind=source_kind,
                source_path=source_path,
                source_records=selected_records,
                staged_viewers=staged_viewers,
                failures=failures,
            )
        except Exception as write_err:
            _log(f"manifest_write_failed: {write_err}")
        return 1


def main() -> int:
    return _run(sys.argv)


if __name__ == "__main__":
    raise SystemExit(main())
