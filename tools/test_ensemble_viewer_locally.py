"""Spin up a local HTTP server in an experiment dir and open the
ensemble_viewer with a synthetic ``?xyz=<rel-url>`` param that
resolves to prism_screen's accepted_ensemble.xyz via relative paths.

Mirrors ``test_geometry_viewer_locally.py``'s shape; differs in:

  * looks for ``outputs/<node_id>/`` dirs whose deployed index.html
    came from the ensemble_viewer generator (auto-detects by reading
    a marker comment in the deployed JS)
  * defaults to prism_screen's ``accepted_ensemble.xyz`` artifact
    instead of xtb's ``xtbopt.xyz``
  * the auto-deploy step pulls assets from
    ``gen_output_node_ensemble_viewer.py`` instead of
    ``gen_output_node_geometry_viewer.py``

Usage::

    python tools/test_ensemble_viewer_locally.py \\
        --experiment-dir /path/to/downloaded/7955 \\
        [--node-id 105104] \\
        [--upstream-protocol-id 105041] \\
        [--upstream-protocol-name wf_prism_imported] \\
        [--file-name accepted_ensemble.xyz] \\
        [--port 8766]

Auto-detect picks the latest prism call dir (highest numeric suffix)
and its ``accepted_ensemble.xyz`` when the optional flags are
omitted, so the common case is just::

    python tools/test_ensemble_viewer_locally.py --experiment-dir ~/Downloads/7955

When the GUI is reachable again you don't need this; it's purely for
offline iteration on the viewer JS.
"""

from __future__ import annotations

import argparse
import http.server
import importlib.util
import json
import os
import shutil
import socketserver
import sys
import urllib.parse
import webbrowser
from pathlib import Path


# Marker the ensemble_viewer's generated viewer.js starts with —
# distinguishes its deployment dirs from geometry_viewer's.
ENSEMBLE_MARKER = "// ensemble_viewer"


def _auto_detect_node(experiment_dir: Path) -> tuple[int, str]:
    """Find the first output node dir whose deployed JS is an
    ensemble_viewer (matched by the leading comment marker).
    """
    outputs = experiment_dir / "outputs"
    if not outputs.is_dir():
        raise SystemExit(f"no outputs/ dir under {experiment_dir}")

    candidates: list[tuple[int, str]] = []
    for node_dir in sorted(p for p in outputs.iterdir() if p.is_dir() and p.name.isdigit()):
        for block_dir in sorted(p for p in node_dir.iterdir()
                                if p.is_dir() and p.name.startswith("onlb_")):
            js = block_dir / "js" / "viewer.js"
            html = block_dir / "index.html"
            if not html.is_file():
                continue
            if js.is_file():
                head = js.read_text(encoding="utf-8", errors="replace")[:256]
                if ENSEMBLE_MARKER in head:
                    return int(node_dir.name), block_dir.name
            candidates.append((int(node_dir.name), block_dir.name))

    # No JS marker hit — fall back to the first available output node
    # and rely on the auto-deploy step to overwrite its viewer.js with
    # the ensemble version. Helpful when the experiment was run with
    # an older generator before the marker existed.
    if candidates:
        nid, blk = candidates[0]
        print(f"[warn] no ensemble_viewer marker found under "
              f"outputs/; falling back to first node: {nid}/{blk}. "
              f"The deploy step will overwrite it with the ensemble JS.")
        return nid, blk
    raise SystemExit(f"no output node deployments under {outputs}")


def _auto_detect_upstream_prism(experiment_dir: Path) -> tuple[int, str, str]:
    """Find the latest wf_prism_imported_<N> call dir and its
    accepted_ensemble.xyz. Latest = highest protocol_id, which in
    practice maps to the second prism pass in a CREST→prism→DFT→prism
    pipeline.
    """
    calls = experiment_dir / "calls"
    if not calls.is_dir():
        raise SystemExit(f"no calls/ dir under {experiment_dir}")
    prism_dirs = sorted(
        (p for p in calls.iterdir() if p.name.startswith("wf_prism_imported_")),
        key=lambda p: int(p.name.rpartition("_")[-1]),
    )
    if not prism_dirs:
        raise SystemExit(
            f"no wf_prism_imported_* call dirs under {calls} — "
            "pass --upstream-protocol-id / --upstream-protocol-name "
            "explicitly."
        )
    prism = prism_dirs[-1]  # latest
    name, _, pid = prism.name.rpartition("_")
    xyz_path = prism / "outputs" / "accepted_ensemble.xyz"
    if not xyz_path.is_file():
        raise SystemExit(
            f"no accepted_ensemble.xyz under {prism}/outputs/ — "
            "pass --file-name explicitly."
        )
    return int(pid), name, "accepted_ensemble.xyz"


def _deploy_latest_viewer_assets(experiment_dir: Path, node_id: int, block_key: str) -> None:
    """Overwrite the deployed viewer files with the source-of-truth
    strings from ``gen_output_node_ensemble_viewer.py``. Same
    rationale as the L1 harness: keeps deployment in sync with the
    generator without forcing repackage.
    """
    gen_path = Path(__file__).resolve().parent / "gen_output_node_ensemble_viewer.py"
    if not gen_path.is_file():
        print(f"[warn] could not find {gen_path} — skipping deploy step")
        return
    spec = importlib.util.spec_from_file_location("_gen_ensemble", gen_path)
    gen = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(gen)

    deploy_dir = experiment_dir / "outputs" / str(node_id) / block_key
    if not deploy_dir.is_dir():
        print(f"[warn] deploy dir {deploy_dir} doesn't exist — skipping")
        return
    (deploy_dir / "index.html").write_text(gen.INDEX_HTML, encoding="utf-8")
    (deploy_dir / "js").mkdir(parents=True, exist_ok=True)
    (deploy_dir / "js" / "viewer.js").write_text(gen.VIEWER_JS, encoding="utf-8")
    (deploy_dir / "css").mkdir(parents=True, exist_ok=True)
    (deploy_dir / "css" / "styles.css").write_text(gen.STYLES_CSS, encoding="utf-8")
    print(f"[deploy] refreshed ensemble_viewer assets in {deploy_dir}")


def _build_viewer_url(
    *, host: str, port: int,
    node_id: int, block_key: str,
    upstream_pid: int, upstream_name: str, file_name: str,
) -> str:
    rel_xyz = (
        f"../../../calls/{upstream_name}_{upstream_pid}/outputs/{file_name}"
    )
    qs = urllib.parse.urlencode({"xyz": rel_xyz})
    return (
        f"http://{host}:{port}/outputs/{node_id}/{block_key}/index.html?{qs}"
    )


def _auto_detect_smiles(experiment_dir: Path) -> str | None:
    """Walk every manifest.json under ``<exp>/calls/`` looking for an
    ``inputs.smiles`` value. Returns the first non-empty hit, or
    ``None`` if nothing matches. Used to surface the molecule's SMILES
    on the viewer sidebar without forcing the user to pass --smiles.
    """
    calls = experiment_dir / "calls"
    if not calls.is_dir():
        return None
    # smiles_to_3d's manifest is the most likely carrier (it's the
    # node that *requires* SMILES), so prefer it if present.
    candidates = sorted(calls.glob("wf_smiles_to_3d*/outputs/manifest.json"))
    candidates += sorted(calls.glob("wf_embed_*/outputs/manifest.json"))
    candidates += sorted(calls.glob("*/outputs/manifest.json"))
    seen: set[Path] = set()
    for mf in candidates:
        if mf in seen:
            continue
        seen.add(mf)
        try:
            data = json.loads(mf.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        smi = (data.get("inputs") or {}).get("smiles")
        if smi:
            return str(smi).strip()
    return None


def _stage_xyz_file(experiment_dir: Path, src: Path) -> str:
    """Copy an arbitrary xyz file into a known location under the
    experiment dir so the local server can serve it as a sibling
    of the viewer. Returns the relative URL the viewer should fetch.

    Staging goes under ``<exp>/.harness_staged/`` to keep it
    visually separate from real experiment artifacts. Pre-existing
    staged files are explicitly unlinked before the new copy lands —
    ``shutil.copy2`` fails with PermissionError if the destination is
    read-only or carries restrictive macOS attributes (seen in
    practice when Downloads-hosted files inherit quarantine flags
    between runs).
    """
    if not src.is_file():
        raise SystemExit(f"--xyz-file not found: {src}")
    staged_dir = experiment_dir / ".harness_staged"
    staged_dir.mkdir(exist_ok=True)
    dest = staged_dir / src.name
    # Defensive cleanup of any stale staged file. Only catch
    # OSError here — bubble unexpected failures up.
    if dest.exists():
        try:
            dest.unlink()
        except OSError as e:
            print(f"[warn] could not remove existing {dest}: {e}")
            print(f"[warn] try: rm -rf {staged_dir}")
            raise
    shutil.copy2(src, dest)
    print(f"[stage] copied {src} → {dest}")
    # Path is relative to viewer's index.html which sits under
    # outputs/<node>/<block>/, so back up three dirs to reach <exp>.
    return f"../../../.harness_staged/{urllib.parse.quote(src.name)}"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Local-server smoke test for the ensemble_viewer node."
    )
    ap.add_argument("--experiment-dir", type=Path, required=True)
    ap.add_argument("--node-id", type=int, default=None)
    ap.add_argument("--upstream-protocol-id", type=int, default=None)
    ap.add_argument("--upstream-protocol-name", default=None)
    ap.add_argument("--file-name", default=None)
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--no-deploy", action="store_true",
                    help="Skip overwriting the experiment dir's deployed "
                         "viewer files with the latest source.")
    ap.add_argument(
        "--xyz-file", type=Path, default=None,
        help="Render an arbitrary multi-frame xyz file instead of "
             "auto-detecting one from the experiment dir's prism "
             "calls. The file is copied under "
             "<experiment-dir>/.harness_staged/ so the local server "
             "can serve it as a sibling of the viewer. Useful for "
             "spot-checking xyzs from outside the experiment "
             "(uploaded by users, scraped from CREST runs, etc.).",
    )
    ap.add_argument(
        "--smiles", default=None,
        help="SMILES string for the molecule. Surfaced on the viewer "
             "sidebar (with double-click-to-copy) and used to render "
             "the 2D structure inset in the viewer's bottom-left "
             "corner. If omitted, the harness scans the experiment "
             "dir's manifests for an upstream node's inputs.smiles.",
    )
    ap.add_argument(
        "--no-smiles", action="store_true",
        help="Explicitly suppress SMILES auto-detection. The sidebar "
             "won't show the SMILES block and the 2D inset stays "
             "hidden. Use this if the auto-detected SMILES is wrong.",
    )
    args = ap.parse_args()

    exp = args.experiment_dir.resolve()
    if not exp.is_dir():
        raise SystemExit(f"not a directory: {exp}")

    if args.node_id is None:
        node_id, block_key = _auto_detect_node(exp)
    else:
        node_id = args.node_id
        node_dir = exp / "outputs" / str(node_id)
        block_dirs = sorted(
            p for p in node_dir.iterdir()
            if p.is_dir() and p.name.startswith("onlb_")
        )
        if not block_dirs:
            raise SystemExit(f"no onlb_* under {node_dir}")
        block_key = block_dirs[0].name

    if not args.no_deploy:
        _deploy_latest_viewer_assets(exp, node_id, block_key)

    # SMILES resolution: explicit --smiles wins, then auto-detect from
    # the experiment dir's manifests, unless --no-smiles is set.
    if args.no_smiles:
        smiles = None
    elif args.smiles is not None:
        smiles = args.smiles.strip() or None
    else:
        smiles = _auto_detect_smiles(exp)
        if smiles:
            print(f"[smiles] auto-detected: {smiles}")
        else:
            print("[smiles] none found in experiment manifests")

    if args.xyz_file is not None:
        # User-supplied xyz: stage it and build the URL directly.
        rel_xyz = _stage_xyz_file(exp, args.xyz_file.resolve())
        qs_pairs = {"xyz": rel_xyz}
        if smiles:
            qs_pairs["smiles"] = smiles
        qs = urllib.parse.urlencode(qs_pairs, quote_via=urllib.parse.quote)
        url = (
            f"http://localhost:{args.port}/outputs/{node_id}/{block_key}/"
            f"index.html?{qs}"
        )
    else:
        # Default: auto-detect the latest prism call.
        if (
            args.upstream_protocol_id is None
            or args.upstream_protocol_name is None
            or args.file_name is None
        ):
            auto_pid, auto_name, auto_file = _auto_detect_upstream_prism(exp)
            upstream_pid = args.upstream_protocol_id or auto_pid
            upstream_name = args.upstream_protocol_name or auto_name
            file_name = args.file_name or auto_file
        else:
            upstream_pid = args.upstream_protocol_id
            upstream_name = args.upstream_protocol_name
            file_name = args.file_name

        url = _build_viewer_url(
            host="localhost", port=args.port,
            node_id=node_id, block_key=block_key,
            upstream_pid=upstream_pid, upstream_name=upstream_name,
            file_name=file_name,
        )
        if smiles:
            sep = "&" if "?" in url else "?"
            url += f"{sep}smiles={urllib.parse.quote(smiles)}"

    print(f"Serving {exp} on http://localhost:{args.port}")
    print(f"Viewer URL:\n  {url}")
    print()
    print("Click conformer cards in the sidebar to switch the active one,")
    print("toggle View mode for focus-vs-overlay, Ctrl-C to stop.")

    if not args.no_browser:
        webbrowser.open(url)

    os.chdir(exp)
    handler = http.server.SimpleHTTPRequestHandler
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", args.port), handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nshutting down")
            return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
