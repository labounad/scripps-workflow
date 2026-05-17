"""Spin up a local HTTP server in an experiment dir and open the
geometry_viewer with synthetic query params that resolve to a real
xyz file via relative paths.

This exercises the viewer's render pipeline end-to-end (CDN load,
fetch, 3Dmol render) without needing the workflow.scripps.edu GUI
to be reachable. The only thing it doesn't exercise is the
exp_services.php round-trip, since the local server doesn't speak
that API; instead the harness uses the viewer's ``LOCAL_TEST`` mode
(``?xyz=<relative-url>``) to bypass the service layer entirely.

Usage:

    python tools/test_geometry_viewer_locally.py \\
        --experiment-dir /path/to/downloaded/7955 \\
        [--node-id 105103] \\
        [--upstream-protocol-id 105037] \\
        [--upstream-protocol-name wf_xtb_imported] \\
        [--file-name xtbopt.xyz] \\
        [--port 8765]

The harness auto-detects sensible defaults from the experiment dir
when the optional flags are omitted, so the common case is just::

    python tools/test_geometry_viewer_locally.py --experiment-dir ~/Downloads/7955

It will print the URL to open and start serving. ctrl-C to stop.

What it actually does:
    1. Locate the output-node deployment under
       ``<exp>/outputs/<node_id>/<block_key>/index.html``.
    2. Locate the upstream xyz under
       ``<exp>/calls/<protocol_name>_<protocol_id>/outputs/<file_name>``.
    3. Compute the relative URL from the viewer to that xyz.
    4. Start ``http.server`` rooted at the experiment dir.
    5. Open
       ``http://localhost:<port>/outputs/<node_id>/<block_key>/index.html
         ?xyz=<encoded-relative-url>``
       which the viewer's LOCAL_TEST mode fetches directly.

When the GUI is reachable again, you don't need this script; use it
only for offline iteration.
"""

from __future__ import annotations

import argparse
import http.server
import importlib.util
import os
import socketserver
import sys
import urllib.parse
import webbrowser
from pathlib import Path


def _auto_detect_node(experiment_dir: Path) -> tuple[int, str]:
    """Pick the first output node deployed under ``<exp>/outputs/``."""
    outputs = experiment_dir / "outputs"
    if not outputs.is_dir():
        raise SystemExit(f"no outputs/ dir under {experiment_dir}")
    node_dirs = sorted(
        p for p in outputs.iterdir()
        if p.is_dir() and p.name.isdigit()
    )
    if not node_dirs:
        raise SystemExit(f"no numeric node-id dirs under {outputs}")
    node_dir = node_dirs[0]
    block_dirs = sorted(
        p for p in node_dir.iterdir()
        if p.is_dir() and p.name.startswith("onlb_")
    )
    if not block_dirs:
        raise SystemExit(f"no onlb_* block dirs under {node_dir}")
    if not (block_dirs[0] / "index.html").is_file():
        raise SystemExit(f"no index.html in {block_dirs[0]}")
    return int(node_dir.name), block_dirs[0].name


def _auto_detect_upstream_xtb(experiment_dir: Path) -> tuple[int, str, str]:
    """Find the first wf_xtb_imported_<N> call dir and its xtbopt.xyz.

    Returns ``(protocol_id, protocol_name, file_name)``.
    """
    calls = experiment_dir / "calls"
    if not calls.is_dir():
        raise SystemExit(f"no calls/ dir under {experiment_dir}")
    xtb_dirs = sorted(p for p in calls.iterdir() if p.name.startswith("wf_xtb_imported_"))
    if not xtb_dirs:
        raise SystemExit(
            f"no wf_xtb_imported_* call dirs under {calls} — "
            "pass --upstream-protocol-id / --upstream-protocol-name "
            "explicitly."
        )
    xtb = xtb_dirs[0]
    # protocol_id is the trailing integer; protocol_name is everything
    # before the last "_<int>".
    name, _, pid = xtb.name.rpartition("_")
    xyz_path = xtb / "outputs" / "xtbopt.xyz"
    if not xyz_path.is_file():
        # Fall back to input.xyz if xtbopt.xyz is missing (e.g. no
        # optimize ran).
        alt = xtb / "outputs" / "input.xyz"
        if alt.is_file():
            return int(pid), name, "input.xyz"
        raise SystemExit(
            f"no xtbopt.xyz or input.xyz under {xtb}/outputs/ — "
            "pass --file-name explicitly."
        )
    return int(pid), name, "xtbopt.xyz"


def _deploy_latest_viewer_assets(experiment_dir: Path, node_id: int, block_key: str) -> None:
    """Overwrite the deployed viewer files with the source-of-truth
    strings from the sibling ``gen_output_node_geometry_viewer.py``.

    Why: the experiment archive captures whatever viewer.js shipped at
    run time, which goes stale the moment we regenerate the node ZIP.
    Re-deploying on every harness start means we always render the
    latest code without forcing the user to repackage / re-unzip.
    """
    gen_path = Path(__file__).resolve().parent / "gen_output_node_geometry_viewer.py"
    if not gen_path.is_file():
        print(f"[warn] could not find {gen_path} — skipping deploy step")
        return
    spec = importlib.util.spec_from_file_location("_gen", gen_path)
    gen = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(gen)

    deploy_dir = experiment_dir / "outputs" / str(node_id) / block_key
    if not deploy_dir.is_dir():
        print(f"[warn] deploy dir {deploy_dir} doesn't exist — skipping")
        return
    (deploy_dir / "index.html").write_text(gen.INDEX_HTML, encoding="utf-8")
    (deploy_dir / "js" / "viewer.js").parent.mkdir(parents=True, exist_ok=True)
    (deploy_dir / "js" / "viewer.js").write_text(gen.VIEWER_JS, encoding="utf-8")
    (deploy_dir / "css" / "styles.css").parent.mkdir(parents=True, exist_ok=True)
    (deploy_dir / "css" / "styles.css").write_text(gen.STYLES_CSS, encoding="utf-8")
    print(f"[deploy] refreshed viewer assets in {deploy_dir}")


def _build_viewer_url(
    *, host: str, port: int,
    node_id: int, block_key: str,
    upstream_pid: int, upstream_name: str, file_name: str,
) -> str:
    # Path from the viewer (under outputs/<node>/<block>/) to the xtb
    # output (under calls/<name>_<pid>/outputs/<file>).
    rel_xyz = (
        f"../../../calls/{upstream_name}_{upstream_pid}/outputs/{file_name}"
    )
    qs = urllib.parse.urlencode({"xyz": rel_xyz})
    return (
        f"http://{host}:{port}/outputs/{node_id}/{block_key}/index.html?{qs}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Local-server smoke test for the geometry_viewer node."
    )
    ap.add_argument(
        "--experiment-dir", type=Path, required=True,
        help="Path to a downloaded experiment dir (the one containing "
             "outputs/, calls/, scripts/, filelist.json).",
    )
    ap.add_argument(
        "--node-id", type=int, default=None,
        help="Output node id (the numeric dir under outputs/). "
             "Auto-detected if omitted.",
    )
    ap.add_argument(
        "--upstream-protocol-id", type=int, default=None,
        help="Upstream call's protocol_id. Auto-detected if omitted.",
    )
    ap.add_argument(
        "--upstream-protocol-name", default=None,
        help="Upstream call's protocol_name (e.g. wf_xtb_imported). "
             "Auto-detected if omitted.",
    )
    ap.add_argument(
        "--file-name", default=None,
        help="Filename under <upstream call>/outputs/ to render "
             "(e.g. xtbopt.xyz). Auto-detected if omitted.",
    )
    ap.add_argument(
        "--port", type=int, default=8765,
        help="Local HTTP port. Default 8765.",
    )
    ap.add_argument(
        "--no-browser", action="store_true",
        help="Don't auto-open the URL — just print it.",
    )
    ap.add_argument(
        "--no-deploy", action="store_true",
        help="Skip overwriting the experiment dir's deployed viewer "
             "files with the latest source from "
             "gen_output_node_geometry_viewer.py. Use this if you "
             "want to test exactly what the GUI deployed (e.g. to "
             "reproduce a specific historical bug).",
    )
    args = ap.parse_args()

    exp = args.experiment_dir.resolve()
    if not exp.is_dir():
        raise SystemExit(f"not a directory: {exp}")

    node_id, block_key = (
        (args.node_id, _auto_detect_node(exp)[1])
        if args.node_id is not None
        else _auto_detect_node(exp)
    )
    # When --node-id is given but block_key isn't, still resolve block_key
    # under that specific node dir:
    if args.node_id is not None:
        node_dir = exp / "outputs" / str(args.node_id)
        block_dirs = sorted(
            p for p in node_dir.iterdir()
            if p.is_dir() and p.name.startswith("onlb_")
        )
        if not block_dirs:
            raise SystemExit(f"no onlb_* under {node_dir}")
        node_id = args.node_id
        block_key = block_dirs[0].name

    if (
        args.upstream_protocol_id is None
        or args.upstream_protocol_name is None
        or args.file_name is None
    ):
        auto_pid, auto_name, auto_file = _auto_detect_upstream_xtb(exp)
        upstream_pid = args.upstream_protocol_id or auto_pid
        upstream_name = args.upstream_protocol_name or auto_name
        file_name = args.file_name or auto_file
    else:
        upstream_pid = args.upstream_protocol_id
        upstream_name = args.upstream_protocol_name
        file_name = args.file_name

    if not args.no_deploy:
        _deploy_latest_viewer_assets(exp, node_id, block_key)

    url = _build_viewer_url(
        host="localhost", port=args.port,
        node_id=node_id, block_key=block_key,
        upstream_pid=upstream_pid, upstream_name=upstream_name,
        file_name=file_name,
    )

    print(f"Serving {exp} on http://localhost:{args.port}")
    print(f"Viewer URL:\n  {url}")
    print()
    print("Click around, switch styles, etc. Ctrl-C to stop.")

    if not args.no_browser:
        webbrowser.open(url)

    os.chdir(exp)
    handler = http.server.SimpleHTTPRequestHandler
    # Allow reusing a recently-vacated port without TIME_WAIT delay.
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
