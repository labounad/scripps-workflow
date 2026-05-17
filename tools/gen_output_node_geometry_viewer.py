"""Generate a GUI Output node ZIP — 3D molecular geometry viewer.

First node of subtype ``Layout`` (Output category). Renders a single
xyz file in an interactive 3Dmol.js viewer with a style selector.
Intended to consume :mod:`scripps_workflow.nodes.xtb_calc`'s
optimized geometry (published under the ``xyz`` artifact bucket by
the ``wf-xtb`` node), but works on any well-formed xyz produced
upstream — it doesn't care what produced the file.

ZIP layout (matches the fasta_viewer reference exactly):

    <node_id>/<block_key>/index.html       <- entry_point
    <node_id>/<block_key>/js/viewer.js
    <node_id>/<block_key>/css/styles.css
    <node_name>.json                       <- node manifest

Runtime contract (mirrors fasta_viewer.js):

    1. The GUI loads index.html inside an iframe with query params
       ``?protocol_id=&user_id=&experiment_id=`` set.
    2. The page calls
       ``opaat.scripps.edu/workflow_webapp_api/services.php?
         serviceName=get_inputs_for_output_node&...&index=0``
       to resolve inport 0 (``xyz_file``) to a concrete resource URL.
    3. The page fetches that resource as text (the xyz file).
    4. The xyz is loaded into a 3Dmol.js viewer; the user can pick
       ball-and-stick / stick / sphere / line and reset view.

3Dmol.js is pulled from the CDN at https://3Dmol.org/build/3Dmol-min.js
— no bundling, no offline mode. The fasta_viewer reference does the
same with proseqviewer.css, so the workflow GUI host accepts external
CDN loads.

After this node ships, the same generator can be cloned for the
next two Layout outputs in the v6 plan: a conformer-ensemble viewer
(multi-model 3Dmol load + ΔE sidebar) and an NMR-labeled diagram
(SVG with a divergent-palette shift gradient). The HTML + JS
template here is intentionally generic enough that cloning it for
those is a 20-minute job.

Usage::

    python tools/gen_output_node_geometry_viewer.py
    # writes ./new_nodes_output/NODE_geometry_viewer_<hash>.zip

Tweak the constants at the top of the script for the author block
and the CDN URL.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import zipfile
from pathlib import Path


OUT_DIR = Path("new_nodes_output")

# Placeholder node_id — the GUI re-assigns this at import time. Pick
# anything stable; the ZIP just needs internal consistency between
# the ``<node_id>/`` directory prefix and the manifest's ``node_id``.
PLACEHOLDER_NODE_ID = 999

NODE_NAME = "geometry_viewer"
NODE_DESCRIPTION = (
    "3D molecular geometry viewer (3Dmol.js). Consumes one xyz file "
    "from an upstream node (typically wf-xtb-calc's optimized "
    "geometry) and renders it as an interactive ball-and-stick / "
    "stick / sphere / line model. Click-drag to rotate, scroll to "
    "zoom, the style selector in the top toolbar switches "
    "representations."
)
NODE_VERSION = "1.0.0"

# Match the author profile used by the v5/v6/v7 input + tag generators
# so re-uploaded ZIPs land under one identity in the GUI's node browser.
AUTHOR = {
    "user_id": 102,
    "name": "Lucas",
    "lastname": "Abounader",
    "lab": "Shenvi",
    "email": "labounader@scripps.edu",
    "main_role": "Designer",
    "first_connection": "2026-02-13 21:09:00",
    "last_connection": "2026-05-16 12:00:00",
    "isAzureUser": 0,
}

# 3Dmol.js CDN. Pinned to the unversioned ``build/`` URL the project
# itself documents — they refresh it in place rather than ship tagged
# releases. If a future build breaks our use we can fall back to a
# cdnjs-pinned version (e.g. 2.0.4). The page degrades gracefully:
# load failures surface a "couldn't load 3Dmol.js" banner instead of
# a blank viewer.
THREEDMOL_CDN = "https://3Dmol.org/build/3Dmol-min.js"

# opaat.scripps.edu service the workflow GUI uses to resolve an
# output node's inport bindings. Matches fasta_viewer.js verbatim;
# changing this requires coordinating with the workflow GUI team.
SERVICES_BASE = (
    "http://opaat.scripps.edu/workflow_webapp_api/services.php"
    "?serviceName=get_inputs_for_output_node"
)
RESOURCE_URL_PREFIX = "http://opaat.scripps.edu/"


# ---------------------------------------------------------------------------
# File contents
# ---------------------------------------------------------------------------


INDEX_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>Geometry Viewer</title>
    <link rel="stylesheet" type="text/css" href="css/styles.css">
    <script src="{cdn}"></script>
    <script src="js/viewer.js"></script>
</head>
<body onload="load_page()">
    <div id="toolbar">
        <label for="style-select">Style:</label>
        <select id="style-select" onchange="apply_style(this.value)">
            <option value="ballstick" selected>Ball &amp; stick</option>
            <option value="stick">Stick</option>
            <option value="sphere">Sphere (CPK)</option>
            <option value="line">Line</option>
        </select>
        <button type="button" onclick="reset_view()">Reset view</button>
        <span id="status">Loading...</span>
    </div>
    <div id="viewer"></div>
    <!-- Drag-and-drop overlay shown only when no upstream binding can
         be resolved (file:// with no params, or a not-yet-released
         GUI integration). Lets the user drop an xyz file and render
         it inline — useful for offline spot-checks. -->
    <div id="dropzone" hidden>
        <div class="dropzone-inner">
            <strong>No upstream xyz bound.</strong><br>
            Drop an <code>.xyz</code> file here, or
            <label class="filepicker">
                <input type="file" accept=".xyz,chemical/x-xyz" onchange="open_picked_file(this)">
                pick a file
            </label>
            to render.
        </div>
    </div>
</body>
</html>
""".format(cdn=THREEDMOL_CDN)


VIEWER_JS = """\
// geometry_viewer — 3Dmol.js renderer for an upstream xyz file.
//
// Resolution modes (first one that has all required URL params wins):
//
//   A. LOCAL_TEST: ?xyz=<absolute-or-relative-url>
//      Direct fetch of the supplied URL. Used by
//      ``tools/test_geometry_viewer_locally.py`` to exercise the
//      render pipeline against an experiment dir served via
//      ``python -m http.server`` (no GUI required).
//
//   B. WORKFLOW_GUI: ?experiment_id=&protocol_id=&protocol_name=&file_name=
//      Real workflow.scripps.edu integration. Calls the backend's
//      ``exp_services.php?serviceName=download_file_to_server`` with
//      the same param shape observed in the experiment call.out log,
//      then fetches the resolved URL. This is a best-guess pending
//      first-hand observation of the GUI's actual iframe URL.
//
//   C. OPAAT_LEGACY: ?protocol_id=&user_id=
//      The fasta_viewer-reference pattern at opaat.scripps.edu. Kept
//      as a fallback in case some output is served from there.
//
//   D. NONE: file://, or no resolvable params.
//      Shows a drag-and-drop UI. User drops an xyz file; we load it
//      via FileReader. Useful for offline xyz spot-checks.
//
// The viewer instance is stashed on window.__viewer so the toolbar
// callbacks (apply_style, reset_view) can reach back into it without
// passing state through DOM attributes.

// Real-GUI service (workflow.scripps.edu). Inferred from the URL
// pattern in the experiment's call.out — confirm against an actual
// iframe load when the GUI is reachable.
var WORKFLOW_GUI_SERVICE =
    'https://workflow.scripps.edu/workflow_backend/exp_services.php'
    + '?serviceName=download_file_to_server';

// Legacy opaat path — fasta_viewer.js's URL shape. Used only if
// ?user_id is present.
var OPAAT_LEGACY_SERVICE = '__OPAAT_SERVICE__';
var OPAAT_RESOURCE_PREFIX = '__OPAAT_RESOURCE_PREFIX__';

function extract_get_parameters() {
    return new URLSearchParams(window.location.search);
}

function set_status(text, is_error) {
    var el = document.getElementById('status');
    if (!el) return;
    el.textContent = text;
    el.style.color = is_error ? '#b22222' : '#333';
}

function load_page() {
    if (typeof $3Dmol === 'undefined') {
        set_status('Failed to load 3Dmol.js from CDN', true);
        return;
    }
    var params = extract_get_parameters();
    var mode = pick_resolution_mode(params);
    console.log('geometry_viewer resolution mode:', mode);
    switch (mode) {
        case 'LOCAL_TEST':    return resolve_local_test(params);
        case 'WORKFLOW_GUI':  return resolve_workflow_gui(params);
        case 'OPAAT_LEGACY':  return resolve_opaat_legacy(params);
        case 'NONE':
        default:              return show_dropzone();
    }
}

function pick_resolution_mode(params) {
    if (params.get('xyz')) return 'LOCAL_TEST';
    if (params.get('experiment_id') && params.get('protocol_id') &&
        params.get('protocol_name') && params.get('file_name')) {
        return 'WORKFLOW_GUI';
    }
    if (params.get('protocol_id') && params.get('user_id')) {
        return 'OPAAT_LEGACY';
    }
    return 'NONE';
}

// ---- Mode A: direct-URL fetch ----------------------------------------

function resolve_local_test(params) {
    var url = params.get('xyz');
    set_status('local-test mode: fetching ' + url);
    fetch_xyz(url);
}

// ---- Mode B: real GUI integration (workflow.scripps.edu) -------------

function resolve_workflow_gui(params) {
    var url = WORKFLOW_GUI_SERVICE
        + '&experiment_id=' + encodeURIComponent(params.get('experiment_id'))
        + '&protocol_id='   + encodeURIComponent(params.get('protocol_id'))
        + '&protocol_name=' + encodeURIComponent(params.get('protocol_name'))
        + '&file_name='     + encodeURIComponent(params.get('file_name'));
    // podid is in the call.out URL pattern but its purpose isn't
    // documented — forward it through when present, leave it off
    // otherwise.
    if (params.get('podid')) {
        url += '&podid=' + encodeURIComponent(params.get('podid'));
    }
    set_status('workflow-gui mode: resolving via exp_services.php');
    fetch_xyz(url);
}

// ---- Mode C: legacy opaat path ---------------------------------------

function resolve_opaat_legacy(params) {
    var url = OPAAT_LEGACY_SERVICE
        + '&protocol_id=' + encodeURIComponent(params.get('protocol_id'))
        + '&user_id='     + encodeURIComponent(params.get('user_id'))
        + '&index=0';
    set_status('opaat-legacy mode: resolving inport');
    fetch(url)
        .then(function(r) { return r.json(); })
        .then(function(data) {
            if (!data || !data[0] || !data[0].resource_url) {
                set_status('Upstream xyz not yet available', true);
                return;
            }
            var resource = data[0].resource_url.replace('/', '//');
            fetch_xyz(OPAAT_RESOURCE_PREFIX + resource);
        })
        .catch(function(err) {
            console.error(err);
            set_status('Error resolving inport: ' + err, true);
        });
}

// ---- Mode D: drag-and-drop fallback ----------------------------------

function show_dropzone() {
    set_status('no upstream binding — drop a file');
    var dz = document.getElementById('dropzone');
    if (!dz) return;
    dz.hidden = false;
    ['dragenter', 'dragover'].forEach(function(evt) {
        dz.addEventListener(evt, function(e) {
            e.preventDefault();
            e.stopPropagation();
            dz.classList.add('dragging');
        });
    });
    ['dragleave', 'drop'].forEach(function(evt) {
        dz.addEventListener(evt, function(e) {
            e.preventDefault();
            e.stopPropagation();
            dz.classList.remove('dragging');
        });
    });
    dz.addEventListener('drop', function(e) {
        var files = e.dataTransfer && e.dataTransfer.files;
        if (files && files[0]) open_file(files[0]);
    });
}

function open_picked_file(input_el) {
    var f = input_el.files && input_el.files[0];
    if (f) open_file(f);
}

function open_file(file) {
    set_status('reading ' + file.name);
    var reader = new FileReader();
    reader.onload = function(e) {
        document.getElementById('dropzone').hidden = true;
        init_viewer(e.target.result);
    };
    reader.onerror = function() {
        set_status('Error reading file: ' + reader.error, true);
    };
    reader.readAsText(file);
}

// ---- Common: fetch + render -----------------------------------------

function fetch_xyz(url) {
    fetch(url)
        .then(function(response) {
            if (!response.ok) throw new Error('HTTP ' + response.status);
            return response.text();
        })
        .then(function(text) { init_viewer(text); })
        .catch(function(err) {
            console.error(err);
            set_status('Error fetching xyz: ' + err, true);
        });
}

function init_viewer(xyz_text) {
    var element = document.getElementById('viewer');
    var config = { backgroundColor: 'white' };
    var viewer = $3Dmol.createViewer(element, config);
    viewer.addModel(xyz_text, 'xyz');
    viewer.setStyle({}, build_style_spec('ballstick'));
    viewer.zoomTo();
    viewer.render();
    window.__viewer = viewer;
    set_status('');
}

function build_style_spec(style_name) {
    switch (style_name) {
        case 'stick':
            return { stick: { radius: 0.15 } };
        case 'sphere':
            // CPK: each atom rendered at its van der Waals radius.
            return { sphere: { scale: 1.0 } };
        case 'line':
            return { line: { linewidth: 2.0 } };
        case 'ballstick':
        default:
            // Default: thin sticks + half-vdW spheres on atoms.
            return {
                stick: { radius: 0.15 },
                sphere: { scale: 0.25 }
            };
    }
}

function apply_style(style_name) {
    if (!window.__viewer) return;
    window.__viewer.setStyle({}, build_style_spec(style_name));
    window.__viewer.render();
}

function reset_view() {
    if (!window.__viewer) return;
    window.__viewer.zoomTo();
    window.__viewer.render();
}
""".replace("__OPAAT_SERVICE__", SERVICES_BASE).replace(
    "__OPAAT_RESOURCE_PREFIX__", RESOURCE_URL_PREFIX
)


STYLES_CSS = """\
/* geometry_viewer — minimal layout. The GUI hosts us in an iframe
   with its own chrome, so we only style the toolbar and viewer
   container.  3Dmol's canvas fills #viewer at 100% / 100%. */

html, body {
    margin: 0;
    padding: 0;
    height: 100%;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
}

#toolbar {
    padding: 8px 12px;
    background: #f4f4f6;
    border-bottom: 1px solid #d8d8dc;
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 14px;
}

#toolbar label {
    color: #333;
}

#toolbar select,
#toolbar button {
    padding: 4px 8px;
    font-size: 14px;
}

#status {
    margin-left: auto;
    color: #555;
    font-size: 13px;
    font-style: italic;
}

#viewer {
    position: relative;
    width: 100%;
    /* Viewport height minus the toolbar's ~36px (padding included). */
    height: calc(100vh - 36px);
}

/* Drag-and-drop overlay shown when no upstream xyz can be resolved.
   Positioned absolutely on top of #viewer; 3Dmol's canvas (when one
   does eventually load) covers it. */
#dropzone {
    position: absolute;
    top: 36px;
    left: 0;
    right: 0;
    bottom: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(248, 248, 252, 0.92);
    pointer-events: auto;
}
/* The HTML5 ``hidden`` global attribute relies on the UA's
   ``display: none`` rule. Our ``#dropzone { display: flex; }`` has
   higher specificity (ID > attribute) and would otherwise win,
   leaving the overlay permanently visible on top of the viewer.
   This explicit override re-establishes hidden-respect. */
#dropzone[hidden] {
    display: none;
}
#dropzone.dragging {
    background: rgba(220, 232, 255, 0.92);
    outline: 2px dashed #5e8fd8;
    outline-offset: -8px;
}
.dropzone-inner {
    text-align: center;
    color: #333;
    font-size: 15px;
    line-height: 1.6;
    padding: 32px 40px;
    border: 1px dashed #b8b8c0;
    border-radius: 8px;
    background: #fff;
}
.dropzone-inner code {
    background: #efeff3;
    padding: 1px 4px;
    border-radius: 3px;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
}
.filepicker {
    color: #2152a4;
    cursor: pointer;
    text-decoration: underline;
}
.filepicker input {
    display: none;
}
"""


# ---------------------------------------------------------------------------
# Manifest builder
# ---------------------------------------------------------------------------


def _make_block_key() -> str:
    """Produce a fasta_viewer-shaped block_key: ``onlb_<14hex>.<8digits>``."""
    hex_part = secrets.token_hex(7)  # 14 hex chars
    num_part = secrets.randbelow(10**8)
    return f"onlb_{hex_part}.{num_part:08d}"


def _make_file_entry(
    file_info_id: int,
    node_id: int,
    block_key: str,
    file_name: str,
    file_format: str,
    rel_path: str,
    entry_point: bool,
) -> dict:
    """Build one file entry for files_info / layout.Blocks.Files."""
    return {
        "node_files_info_id": file_info_id,
        "node_id": node_id,
        "file_name": file_name,
        "file_format": file_format,
        "upload_method": "code",
        "path": rel_path,
        # The next four fields are duplicated under layout.Blocks.Files;
        # we build the layout side by copying these + adding the keys
        # layout needs (screed, col, block_key, etc.).
    }


def _make_layout_file_entry(
    file_info_id: int,
    node_id: int,
    block_key: str,
    file_name: str,
    file_format: str,
    rel_path: str,
    entry_point: bool,
) -> dict:
    """Build one file entry shaped for ``layout.Blocks[].Files[]``.

    Same fields as files_info plus the grid placement fields the
    layout subtype needs (screed/col/block_key/output_file_name).
    """
    return {
        "node_files_info_id": file_info_id,
        "node_id": node_id,
        "screed": 0,
        "col": 0,
        "block_key": block_key,
        "entry_point": entry_point,
        "output_file_name": "index.webapp",
        "output_file_type": "webapp",
        "file_name": file_name,
        "file_format": file_format,
        "upload_method": "code",
        "path": rel_path,
        "screeds": 1,
        "cols": 1,
    }


def build_manifest(node_id: int, block_key: str) -> dict:
    """Construct the full <node_name>.json manifest.

    Mirrors the fasta_viewer reference's field set exactly; the GUI
    rejects manifests missing any of these top-level keys. New keys
    can be added but should default to falsy values (the GUI's
    schema is permissive on superset, strict on subset).
    """
    base_dir = f"{node_id}/{block_key}"
    files_meta = [
        (1, "index.html",  "unknown",     f"{base_dir}/index.html",       True),
        (2, "viewer.js",   "javascript",  f"{base_dir}/js/viewer.js",     False),
        (3, "styles.css",  "txt",         f"{base_dir}/css/styles.css",   False),
    ]

    files_info = [
        _make_file_entry(fid, node_id, block_key, name, fmt, path, entry)
        for (fid, name, fmt, path, entry) in files_meta
    ]
    layout_files = [
        _make_layout_file_entry(fid, node_id, block_key, name, fmt, path, entry)
        for (fid, name, fmt, path, entry) in files_meta
    ]

    return {
        "node_id": node_id,
        "name": NODE_NAME,
        "node_type": "Output",
        "description": NODE_DESCRIPTION,
        "category": "Layout",
        "domain": "public",
        "author": AUTHOR,
        "node_value": None,
        "directives": "None",
        "is_mpi": 0,
        "processing_type": None,
        "custom_value": 0,
        "file_tracking": 0,
        "limit": None,
        "host": "workflow.scripps.edu",
        "version": NODE_VERSION,
        "files_info": files_info,
        # Single inport — the upstream xyz file.
        "inputs": [
            {
                "node_input_id": 1,
                "node_id": node_id,
                "name": "xyz_file",
                "type": "text",
                "tags": None,
                # required=1: there's no point rendering an empty viewer.
                # The GUI surfaces a "missing required input" warning if
                # the upstream port is unbound at protocol-start time.
                "required": 1,
                "new_id": 1,
            },
        ],
        "outputs": [],
        "params": [],
        "layout": {
            "node_id": node_id,
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
                    "node_id": node_id,
                    "output_file_name": "index.webapp",
                    "output_file_type": "webapp",
                    "Files": layout_files,
                },
            ],
        },
    }


# ---------------------------------------------------------------------------
# ZIP packer
# ---------------------------------------------------------------------------


def _short_hash(payload: bytes) -> str:
    """A 13-char hex hash used for the ZIP filename suffix."""
    return hashlib.sha256(payload).hexdigest()[:13]


def write_zip(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    node_id = PLACEHOLDER_NODE_ID
    block_key = _make_block_key()
    manifest = build_manifest(node_id, block_key)
    manifest_bytes = json.dumps(manifest, indent=2).encode("utf-8")

    # Hash matches the v5/v6/v7 pattern: ``NODE_<name>_<13hex>.zip``.
    # Hash is over the manifest only — same node + same file contents
    # round-trip to the same suffix, simpler diffing across regens.
    zip_path = out_dir / f"NODE_{NODE_NAME}_{_short_hash(manifest_bytes)}.zip"

    base_dir = f"{node_id}/{block_key}"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"{base_dir}/index.html", INDEX_HTML)
        z.writestr(f"{base_dir}/js/viewer.js", VIEWER_JS)
        z.writestr(f"{base_dir}/css/styles.css", STYLES_CSS)
        z.writestr(f"{NODE_NAME}.json", manifest_bytes.decode("utf-8"))

    return zip_path


def main() -> None:
    path = write_zip(OUT_DIR)
    print(f"wrote {path}")
    print(f"  size: {path.stat().st_size} bytes")
    print(f"  files: 4 (index.html + viewer.js + styles.css + {NODE_NAME}.json)")
    print()
    print("Upload via the GUI's Node Manager (same flow as the v5/v6/v7 input")
    print(f"bundles). In the workflow editor, wire wf-xtb's ``xyz`` output bucket")
    print(f"→ {NODE_NAME}.xyz_file, then run the protocol. The viewer iframe")
    print("will appear once xtb_calc emits its xyz artifact (the published")
    print("``xtbopt.xyz`` post-optimize, or the input.xyz on a no-optimize run).")


if __name__ == "__main__":
    main()
