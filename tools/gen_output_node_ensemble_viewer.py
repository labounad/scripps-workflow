"""Generate a GUI Output node ZIP — conformer ensemble viewer.

Second node of subtype ``Layout`` (Output category). Consumes a
multi-frame XYZ file (one frame per accepted conformer, energy
embedded in each frame's comment line) and renders all of them in
3Dmol.js with a sidebar listing each conformer's energy relative to
the minimum, in kcal/mol.

Intended upstream binding: :mod:`scripps_workflow.nodes.prism_screen`'s
``xyz_ensemble`` artifact bucket (its ``accepted_ensemble.xyz``
multi-frame file). Also works on any tool that emits multi-frame XYZ
with energies on the comment line — CREST, GOAT, and ORCA's
``$new_job ! ScanTS`` traj files use the same convention
(``Coordinates from … E <hartree>`` or bare hartree).

UI features
-----------

* **3D viewer** with two render modes (focus+overlay vs single
  conformer), four atom styles, and a Reset view button.
* **2D structure box** in the viewer's bottom-left corner — small
  by default, smoothly expands on hover. Rendered with smiles-drawer
  from the operator-supplied SMILES; gracefully omitted when no
  SMILES is provided.
* **Sidebar header (pinned)** with the molecule's SMILES (double-
  click to copy), conformer count, and an energy-unit toggle
  switching the sidebar cards between ΔE kcal/mol (default) and
  absolute E in hartree.
* **Sparkline (pinned)** showing each conformer's ΔE or Boltzmann
  weight % across the ensemble. Active conformer's dot is colored,
  others greyscale. Click a dot to jump to that conformer; toggle
  the y-axis between ΔE and Boltzmann % via a segmented control.
* **Conformer cards (scrollable)** sorted by ΔE ascending; each
  carries a color band (green→red gradient) and a proportional bar.
  Double-click an energy value to copy it to the clipboard.
* **Keyboard navigation** — ↑ / ↓ cycles through conformers in
  ΔE order; the 3D viewer and sidebar follow.

URL parameter resolution (same shape as geometry_viewer)
--------------------------------------------------------

  A. LOCAL_TEST  (?xyz=<url> [&smiles=<str>])     — local harness
  B. WORKFLOW_GUI (?experiment_id=&protocol_id=&...) — real GUI
  C. OPAAT_LEGACY (?protocol_id=&user_id=)        — fasta_viewer style
  D. NONE                                          — drag-and-drop fallback

Usage::

    python tools/gen_output_node_ensemble_viewer.py
    # writes ./new_nodes_output/NODE_ensemble_viewer_<hash>.zip
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
    "Interactive conformer ensemble viewer (3Dmol.js + multi-frame "
    "XYZ). Consumes a single multi-frame XYZ file (typically "
    "prism_screen's ``accepted_ensemble.xyz``) and renders each "
    "conformer alongside a sidebar listing relative energies in "
    "kcal/mol. Toggle between focus-with-overlay and single-"
    "conformer modes from the toolbar; click any sidebar entry to "
    "switch the active conformer. Energies are parsed from each "
    "frame's comment line (ORCA convention: ``E <hartree>``)."
)
NODE_VERSION = "1.0.0"

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

THREEDMOL_CDN = "https://3Dmol.org/build/3Dmol-min.js"

# Legacy opaat service shape, kept as fallback resolver mode C.
OPAAT_SERVICE = (
    "https://opaat.scripps.edu/workflow_webapp_api/services.php"
    "?serviceName=get_inputs_for_output_node"
)
OPAAT_RESOURCE_PREFIX = "https://opaat.scripps.edu/"


# ---------------------------------------------------------------------------
# File contents
# ---------------------------------------------------------------------------


INDEX_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>Conformer Ensemble Viewer</title>
    <link rel="stylesheet" type="text/css" href="css/styles.css">
    <script src="{cdn}"></script>
    <!-- RDKit.js (minimal WASM build): renders the 2D structure inset
         from the operator-supplied SMILES. Unpinned URL (always-
         latest) — a specific @<version> pin had reachability
         problems on unpkg. ``onerror`` surfaces a console log if
         the script 404s or CSP-blocks; the viewer degrades
         gracefully (no 2D inset) in either case. -->
    <script src="https://unpkg.com/@rdkit/rdkit/dist/RDKit_minimal.js"
            onerror="console.warn('RDKit.js CDN load failed (check Network tab); 2D inset will be hidden')"></script>
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
        <label for="mode-select" style="margin-left:12px;">View:</label>
        <select id="mode-select" onchange="set_view_mode(this.value)">
            <option value="overlay" selected>Focus + overlay</option>
            <option value="single">Single conformer</option>
        </select>
        <button type="button" onclick="reset_view()">Reset zoom</button>
        <span id="status">Loading...</span>
    </div>
    <div id="content">
        <div id="viewer">
            <!-- Atom hover readout: small pill in the bottom-right
                 corner of the viewer that surfaces "<elem><serial>"
                 (e.g. "C2", "O11") when the cursor is over an atom
                 of the active conformer. Other-conformer atoms are
                 ignored. Empty / invisible when nothing is hovered. -->
            <div id="atom-hover-info"></div>
            <!-- Click-to-pin readout: bottom-center of the viewer.
                 Set when the user clicks an atom. Independent of the
                 hover readout above — both can be visible at once
                 (hover updates the corner, click pins the center). -->
            <div id="atom-pinned-label"></div>
            <!-- 2D structure inset. Three visual states:
                 (a) corner   — 80×80, bottom-left, default
                 (b) hover    — 240×240, expanded inline on hover
                 (c) centered — large overlay covering the viewer,
                                used for picking substructures
                 The chevron toggles (a)↔(c). The inner svg host
                 receives RDKit's get_svg_with_highlights() output;
                 clicks on atom-* / bond-* SVG classes are caught
                 via event delegation and drive the selection set. -->
            <div id="structure-2d" hidden
                 ondblclick="structure_bg_dblclick(event)">
                <div id="structure-canvas"></div>
                <button type="button" id="structure-expand-btn"
                        title="Expand to center"
                        onclick="toggle_structure_expanded(event)">&#x26F6;</button>
                <div id="structure-toolbar">
                    <span id="structure-selection-count"></span>
                    <button type="button" class="sb-action align"
                            onclick="align_to_selection()">Align</button>
                    <button type="button" class="sb-action reset"
                            onclick="reset_alignment()">Reset</button>
                </div>
            </div>
        </div>
        <aside id="sidebar">
            <!-- SMILES block — hidden by default, shown when the
                 viewer receives a smiles param via URL. Double-click
                 anywhere on the SMILES text to copy. -->
            <div id="smiles-block" hidden>
                <div class="sb-label">
                    SMILES
                    <span class="sb-hint">double-click to copy</span>
                </div>
                <div id="smiles-value"
                     class="copy-target"
                     ondblclick="copy_smiles_dblclick(this)"></div>
            </div>
            <!-- Pinned sidebar header: count + energy-unit toggle. -->
            <div class="sidebar-section sticky-top">
                <div class="sidebar-header">
                    <strong>Conformers</strong>
                    <span id="conformer-count"></span>
                    <div class="spacer"></div>
                    <div class="seg-toggle" id="energy-unit-toggle">
                        <button type="button" data-val="kcal" class="active"
                                onclick="set_energy_unit('kcal')">&Delta;E kcal/mol</button>
                        <button type="button" data-val="hartree"
                                onclick="set_energy_unit('hartree')">E hartree</button>
                    </div>
                </div>
            </div>
            <!-- Pinned sparkline block: chart + y-axis toggle.
                 viewBox 280×100 (2.8:1) keeps circles circular when
                 the SVG scales to fit the sidebar width — earlier
                 ``preserveAspectRatio=none`` was distorting points. -->
            <div class="sidebar-section sticky-below-header">
                <div class="sparkline-row">
                    <svg id="sparkline" viewBox="0 0 280 100"></svg>
                </div>
                <div class="sparkline-toolbar">
                    <div class="seg-toggle small" id="chart-axis-toggle">
                        <button type="button" data-val="energy" class="active"
                                onclick="set_chart_axis('energy')">&Delta;E</button>
                        <button type="button" data-val="boltzmann"
                                onclick="set_chart_axis('boltzmann')">Boltzmann %</button>
                    </div>
                    <span id="sparkline-range"></span>
                </div>
            </div>
            <!-- Scrollable list of conformer cards. -->
            <div id="conformer-list"></div>
        </aside>
    </div>
    <div id="dropzone" hidden>
        <div class="dropzone-inner">
            <strong>No upstream ensemble bound.</strong><br>
            Drop a multi-frame <code>.xyz</code> file here, or
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
// ensemble_viewer — 3Dmol.js multi-conformer viewer.
//
// Resolution modes (same shape as geometry_viewer):
//   A. LOCAL_TEST: ?xyz=<absolute-or-relative-url> [&smiles=<str>]
//   B. WORKFLOW_GUI: ?experiment_id=&protocol_id=&protocol_name=&file_name=
//                   [&smiles=<str>]
//   C. OPAAT_LEGACY: ?protocol_id=&user_id=
//   D. NONE: drag-and-drop fallback
//
// On fetch, the multi-frame XYZ is parsed into per-conformer frames.
// Each frame's comment line is scanned for ORCA-style "E <hartree>"
// or CREST-style bare-number patterns; the parsed energies drive the
// sidebar's ΔE values, the sparkline, and the gradient/bar viz.
//
// View state is held in window.__viewer_state:
//   {
//     viewer, frames, sorted_indices, active_idx,
//     mode, style, smiles, energy_unit, chart_y_axis,
//     max_dE_kcal, boltzmann_weights
//   }
//
// UI callbacks mutate state and call render_sidebar() / repaint_viewer()
// / render_sparkline() / render_2d_structure() as needed.

var WORKFLOW_GUI_SERVICE =
    'https://workflow.scripps.edu/workflow_backend/exp_services.php'
    + '?serviceName=download_file_to_server';

var OPAAT_LEGACY_SERVICE = '__OPAAT_SERVICE__';
var OPAAT_RESOURCE_PREFIX = '__OPAAT_RESOURCE_PREFIX__';

// Hartree → kcal/mol conversion factor. CODATA 2018 value; matches
// the constant used everywhere else in scripps-workflow / nmr-data.
var HARTREE_TO_KCAL_PER_MOL = 627.5094740631;

// kT at 298.15 K, in kcal/mol. Used for Boltzmann weights on the
// sparkline. R * T / 1000 with R in J/(mol·K).
var KT_KCAL_298K = 0.592484;

// ─── AD-HOC ATOM MAPPING for L2 test fixture ────────────────────────
// The 33-conformer crest_ensemble_33conf.xyz under tests/fixtures/
// has its heavy atoms in CREST's reordered form (O at xyz index 0)
// instead of the SMILES parse order RDKit uses for the 2D drawing
// (O at SMILES index 10). Without correction, clicking atom N in
// the 2D SVG would select the wrong physical atom in the XYZ and
// Kabsch would wobble nonsensically.
//
// The mapping below is derived by hand from the first frame's bond
// connectivity. ``map[smiles_idx] = xyz_idx`` — i.e. SMILES atom 10
// (=O) maps to xyz atom 0. Heavy atoms only; H atoms aren't in
// RDKit's mol from a SMILES parse so they aren't clickable anyway.
//
// Production fix (deferred): use RDKit to read the xyz as a mol,
// then call ``query_mol.get_substruct_match(smiles_mol)`` to derive
// this permutation automatically for any SMILES + xyz pair.
var TEST_FIXTURE_ATOM_MAPS = [
    {
        smiles: 'CC/C=C\\\\CC1=C(CCC1=O)C',  // L2 test fixture
        map:    [7, 6, 5, 4, 3, 2, 8, 10, 11, 1, 0, 9],
    },
];

function get_atom_map_for_smiles(smiles) {
    if (!smiles) return null;
    for (var i = 0; i < TEST_FIXTURE_ATOM_MAPS.length; i++) {
        if (TEST_FIXTURE_ATOM_MAPS[i].smiles === smiles) {
            return TEST_FIXTURE_ATOM_MAPS[i].map;
        }
    }
    return null;
}

function extract_get_parameters() {
    return new URLSearchParams(window.location.search);
}

// Helper for building inline SVG elements.
function svg_elem(tag, attrs) {
    var e = document.createElementNS('http://www.w3.org/2000/svg', tag);
    for (var k in attrs) {
        if (attrs[k] !== undefined && attrs[k] !== null) {
            e.setAttribute(k, attrs[k]);
        }
    }
    return e;
}

// Clipboard helper. Brief visual "flash" on the source element so
// the user gets feedback when the silent navigator.clipboard call
// succeeds. Falls back to no-op on browsers without clipboard API.
function copy_value(text, source_el) {
    if (!navigator || !navigator.clipboard) {
        console.warn('Clipboard API unavailable');
        return;
    }
    navigator.clipboard.writeText(String(text)).then(function() {
        if (source_el) {
            source_el.classList.add('flash-copied');
            setTimeout(function() {
                source_el.classList.remove('flash-copied');
            }, 700);
        }
    }).catch(function(err) {
        console.warn('Copy failed:', err);
    });
}

function copy_smiles_dblclick(el) {
    var state = window.__viewer_state;
    if (!state || !state.smiles) return;
    copy_value(state.smiles, el);
}

function try_build_mol_from_xyz_first_frame(state) {
    // Attempt to construct an RDKit mol from the first conformer's
    // raw XYZ text. When this works, the resulting mol has atoms in
    // the EXACT order the xyz file specifies — so a click on
    // atom-N in the SVG resolves to xyz atom N directly, no
    // SMILES→XYZ mapping table needed. RDKit's bond perception is
    // distance-based; usually robust for typical organic molecules.
    //
    // Returns the mol or null. Caller is responsible for ``mol.delete()``.
    if (!window.__rdkit || !state.frames || state.frames.length === 0) {
        return null;
    }
    var frame_text = state.frames[0].text;  // already a self-contained xyz block
    var mol = null;
    try {
        // get_mol's second arg is a JSON details string. Recent RDKit
        // JS builds (2024.x+) accept xyz blocks here; the
        // ``mappedSmilesAsMatching`` / ``useHueckel`` toggles tune
        // bond perception. We keep it default for now.
        mol = window.__rdkit.get_mol(frame_text);
        if (!mol || !mol.is_valid()) {
            if (mol) { try { mol.delete(); } catch (e) {} }
            console.warn('RDKit could not parse xyz block; ' +
                         'falling back to SMILES-based 2D drawing');
            return null;
        }
        // 2D coords are required for drawing — RDKit's xyz parse
        // gives 3D coords. Force 2D layout regeneration.
        try { mol.set_new_coords(false); } catch (e) {
            // Older builds expose this only as set_new_coords() with no arg
            try { mol.set_new_coords(); } catch (e2) {}
        }
        return mol;
    } catch (err) {
        if (mol) { try { mol.delete(); } catch (e) {} }
        console.warn('xyz-mol build failed:', err);
        return null;
    }
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
    console.log('ensemble_viewer resolution mode:', mode);

    // The SMILES is orthogonal to the resolution mode — any mode can
    // optionally carry it via ?smiles=<encoded>. Stash early so it's
    // available when init_ensemble fires later.
    window.__pending_smiles = params.get('smiles') || null;

    install_keyboard_handler();
    // Kick off RDKit.js WASM init in parallel with the xyz fetch.
    // It typically takes ~500ms; render_2d_structure() waits for both
    // RDKit and the viewer state to be ready before drawing the inset.
    kick_off_rdkit();
    switch (mode) {
        case 'LOCAL_TEST':    return resolve_local_test(params);
        case 'WORKFLOW_GUI':  return resolve_workflow_gui(params);
        case 'OPAAT_LEGACY':  return resolve_opaat_legacy(params);
        case 'NONE':
        default:              return show_dropzone();
    }
}

function kick_off_rdkit() {
    // initRDKitModule is published as a global by RDKit_minimal.js.
    // Check both ``window.initRDKitModule`` and the bare global —
    // some script-loading contexts (sandboxed iframes, strict CSP)
    // surface globals differently.
    var initFn = (typeof window !== 'undefined' && window.initRDKitModule)
        || (typeof initRDKitModule !== 'undefined' ? initRDKitModule : null);
    if (typeof initFn !== 'function') {
        console.warn(
            'RDKit.js global not found (initRDKitModule undefined). ' +
            'Likely the CDN script failed to load — check the Network tab ' +
            'for a 404 or CSP block. 2D inset will be hidden.'
        );
        return;
    }
    if (window.__rdkit_initing || window.__rdkit) return;
    window.__rdkit_initing = true;
    initFn().then(function(RDKit) {
        window.__rdkit = RDKit;
        window.__rdkit_initing = false;
        try {
            console.log('RDKit.js ready, version:', RDKit.version());
        } catch (e) { /* version() optional */ }
        // If the viewer state finished first (xyz fetch was fast),
        // it returned without drawing the 2D inset; do it now that
        // RDKit is ready.
        if (window.__viewer_state && window.__viewer_state.smiles) {
            render_2d_structure();
        }
    }).catch(function(err) {
        window.__rdkit_initing = false;
        console.warn('RDKit init failed:', err);
    });
}

function install_keyboard_handler() {
    // ↑ / ← cycle to the previous (lower-energy) conformer.
    // ↓ / → cycle to the next (higher-energy) conformer.
    // Bound on window once, short-circuits if no state or focus is
    // in an editable element.
    window.addEventListener('keydown', function(e) {
        var key = e.key;
        var prev_keys = (key === 'ArrowUp' || key === 'ArrowLeft');
        var next_keys = (key === 'ArrowDown' || key === 'ArrowRight');
        if (!prev_keys && !next_keys) return;
        var t = e.target;
        if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' ||
                  t.tagName === 'SELECT' || t.isContentEditable)) {
            return;
        }
        var state = window.__viewer_state;
        if (!state || !state.sorted_indices ||
            state.sorted_indices.length === 0) return;
        var pos = state.sorted_indices.indexOf(state.active_idx);
        if (pos < 0) return;
        var next = prev_keys ? pos - 1 : pos + 1;
        if (next < 0 || next >= state.sorted_indices.length) return;
        set_active_conformer(state.sorted_indices[next]);
        e.preventDefault();
    });
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

function resolve_local_test(params) {
    var url = params.get('xyz');
    set_status('local-test mode: fetching ' + url);
    fetch_xyz(url);
}

function resolve_workflow_gui(params) {
    var url = WORKFLOW_GUI_SERVICE
        + '&experiment_id=' + encodeURIComponent(params.get('experiment_id'))
        + '&protocol_id='   + encodeURIComponent(params.get('protocol_id'))
        + '&protocol_name=' + encodeURIComponent(params.get('protocol_name'))
        + '&file_name='     + encodeURIComponent(params.get('file_name'));
    if (params.get('podid')) {
        url += '&podid=' + encodeURIComponent(params.get('podid'));
    }
    set_status('workflow-gui mode: resolving via exp_services.php');
    fetch_xyz(url);
}

function resolve_opaat_legacy(params) {
    var url = OPAAT_LEGACY_SERVICE
        + '&protocol_id=' + encodeURIComponent(params.get('protocol_id'))
        + '&user_id='     + encodeURIComponent(params.get('user_id'))
        + '&index=0';
    set_status('opaat-legacy mode: resolving inport');
    fetch(url)
        .then(function(r) {
            if (!r.ok) throw new Error('resolver HTTP ' + r.status);
            return r.json();
        })
        .then(function(data) {
            if (!data || !data[0] || !data[0].resource_url) {
                set_status('Upstream ensemble not yet available', true);
                return;
            }
            var xyz_url = new URL(data[0].resource_url, OPAAT_RESOURCE_PREFIX).href;
            fetch_xyz(xyz_url);
        })
        .catch(function(err) {
            console.error(err);
            set_status('Error resolving inport: ' + err, true);
        });
}

function show_dropzone() {
    set_status('no upstream binding — drop a file');
    var dz = document.getElementById('dropzone');
    if (!dz) return;
    dz.hidden = false;
    ['dragenter', 'dragover'].forEach(function(evt) {
        dz.addEventListener(evt, function(e) {
            e.preventDefault(); e.stopPropagation();
            dz.classList.add('dragging');
        });
    });
    ['dragleave', 'drop'].forEach(function(evt) {
        dz.addEventListener(evt, function(e) {
            e.preventDefault(); e.stopPropagation();
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
        init_ensemble(e.target.result);
    };
    reader.onerror = function() {
        set_status('Error reading file: ' + reader.error, true);
    };
    reader.readAsText(file);
}

function fetch_xyz(url) {
    fetch(url)
        .then(function(response) {
            if (!response.ok) throw new Error('HTTP ' + response.status);
            return response.text();
        })
        .then(function(text) { init_ensemble(text); })
        .catch(function(err) {
            console.error(err);
            set_status('Error fetching xyz: ' + err, true);
        });
}

// ---- Multi-frame XYZ parsing ----------------------------------------

function parse_multi_frame_xyz(text) {
    // Returns [{index, n_atoms, comment, text, energy_hartree}].
    // ``text`` per frame is a fully self-contained xyz block
    // (n_atoms\\ncomment\\natom_lines\\n) suitable for 3Dmol.addModel.
    var frames = [];
    var lines = text.split(/\\r?\\n/);
    var i = 0;
    while (i < lines.length) {
        // Skip leading blank lines between frames.
        while (i < lines.length && lines[i].trim() === '') i++;
        if (i >= lines.length) break;

        var n_atoms = parseInt(lines[i].trim(), 10);
        if (!Number.isFinite(n_atoms) || n_atoms <= 0) {
            console.warn('parse_multi_frame_xyz: bad atom count at line', i, ':', lines[i]);
            i++;
            continue;
        }
        var comment = (i + 1 < lines.length ? lines[i + 1] : '').trim();
        var frame_lines = lines.slice(i, i + 2 + n_atoms);
        if (frame_lines.length < 2 + n_atoms) {
            console.warn('parse_multi_frame_xyz: truncated frame at index', frames.length);
            break;
        }
        var frame_text = frame_lines.join('\\n') + '\\n';

        // ORCA convention: "Coordinates from orca-job orca_opt E -114.299..."
        // CREST writes the energy alone on the line. Match either; fall
        // back to null if neither.
        var energy_hartree = null;
        var m = comment.match(/\\bE\\s+(-?\\d+(?:\\.\\d+)?(?:[eE][+-]?\\d+)?)/);
        if (m) {
            energy_hartree = parseFloat(m[1]);
        } else {
            // CREST-style: just the energy on its own (no "E " prefix).
            var bare = comment.match(/^\\s*(-?\\d+\\.\\d+(?:[eE][+-]?\\d+)?)\\s*$/);
            if (bare) energy_hartree = parseFloat(bare[1]);
        }

        frames.push({
            index: frames.length + 1,
            n_atoms: n_atoms,
            comment: comment,
            text: frame_text,
            energy_hartree: energy_hartree,
        });
        i += 2 + n_atoms;
    }
    return frames;
}

// ---- Ensemble bootstrap --------------------------------------------

function init_ensemble(text) {
    var frames = parse_multi_frame_xyz(text);
    if (frames.length === 0) {
        set_status('No conformers parsed from xyz file', true);
        return;
    }

    // Compute ΔE in kcal/mol for each frame, relative to the global min.
    var energies_h = frames
        .map(function(f) { return f.energy_hartree; })
        .filter(function(x) { return x !== null && isFinite(x); });
    var min_e = energies_h.length ? Math.min.apply(null, energies_h) : null;
    var max_dE = 0;
    frames.forEach(function(f) {
        if (f.energy_hartree !== null && min_e !== null) {
            f.delta_kcal = (f.energy_hartree - min_e) * HARTREE_TO_KCAL_PER_MOL;
            if (f.delta_kcal > max_dE) max_dE = f.delta_kcal;
        } else {
            f.delta_kcal = null;
        }
    });
    // Normalized rank (0..1) for the gradient COLOR band.
    // (Bar LENGTH uses normalized Boltzmann weight — computed below.)
    frames.forEach(function(f) {
        f.norm = (max_dE > 0 && f.delta_kcal !== null)
            ? Math.max(0, Math.min(1, f.delta_kcal / max_dE))
            : 0;
    });

    // Sorted-by-ΔE order: used by sparkline, keyboard nav, sidebar
    // card order. Frames without energy slot to the end.
    var sorted_indices = frames.map(function(_, i) { return i; })
        .sort(function(a, b) {
            var da = frames[a].delta_kcal, db = frames[b].delta_kcal;
            if (da === null && db === null) return a - b;
            if (da === null) return 1;
            if (db === null) return -1;
            return da - db;
        });

    // Boltzmann weights at 298.15 K, indexed by frame index. Also
    // normalize each weight relative to the ensemble's max — so the
    // lowest-energy (highest-weight) conformer reads as 100% on the
    // sidebar bar, and visually-negligible conformers have near-zero
    // bars. This is more chemically meaningful than length∝ΔE.
    var weights = compute_boltzmann_weights(frames);
    var max_weight = weights.length ? Math.max.apply(null, weights) : 0;
    frames.forEach(function(f, i) {
        f.weight = weights[i] || 0;
        f.weight_norm = max_weight > 0 ? (weights[i] / max_weight) : 0;
    });

    // Build the 3Dmol viewer with every frame loaded as a separate
    // model. setStyle({model: N}, ...) controls visibility/appearance
    // per conformer without re-uploading geometry.
    var element = document.getElementById('viewer');
    var viewer = $3Dmol.createViewer(element, {
        backgroundColor: 'white',
        // 3Dmol's setHoverable defaults to a 500ms dwell time before
        // the callback fires. That makes the atom-hover readout feel
        // laggy; we want it to update the instant the cursor enters
        // an atom. Setting via createViewer options for newer builds
        // plus a direct property assignment below for older builds.
        hoverDuration: 0,
    });
    if ('HOVER_DURATION' in viewer) viewer.HOVER_DURATION = 0;
    frames.forEach(function(f) { viewer.addModel(f.text, 'xyz'); });

    // Hover + click registration happens in repaint_viewer() (called
    // immediately below + on every active-conformer change). Without
    // scoping to the active model, the overlay sticks of inactive
    // conformers catch hover/click events first, blocking the active
    // atom beneath. repaint_viewer flips hoverable/clickable on/off
    // per model so picking always lands on the visible-and-active
    // conformer.

    // Initial active = lowest-E conformer = first in sorted_indices.
    var lowest_idx = sorted_indices[0];

    window.__viewer_state = {
        viewer: viewer,
        frames: frames,
        sorted_indices: sorted_indices,
        active_idx: lowest_idx,
        mode: 'overlay',
        style: 'ballstick',
        smiles: window.__pending_smiles || null,
        energy_unit: 'kcal',       // 'kcal' (ΔE) or 'hartree' (abs E)
        chart_y_axis: 'energy',    // 'energy' or 'boltzmann'
        max_dE_kcal: max_dE,
        boltzmann_weights: weights,
        // Substructure-selection + alignment state. Selection is a
        // plain object {atom_idx: true} for fast lookup; we re-build
        // a Set in JS callbacks rather than carrying one in state to
        // keep this trivially serializable.
        selected_atoms: {},
        selection_count: 0,
        is_aligned: false,         // becomes true after align_to_selection()
        is_expanded: false,        // 2D inset corner vs. centered overlay
        // Cache of the original xyz coords for every atom in every
        // frame. Set lazily by ensure_coords_cache() the first time
        // align_to_selection() fires, so init_ensemble stays cheap
        // for the read-only "I just want to look" path.
        original_coords: null,
    };

    render_smiles_block();
    render_2d_structure();
    render_sidebar();
    render_sparkline();
    repaint_viewer();
    viewer.zoomTo();
    viewer.render();
    set_status('');
}

function compute_boltzmann_weights(frames) {
    var exps = frames.map(function(f) {
        if (f.delta_kcal === null || !isFinite(f.delta_kcal)) return 0;
        return Math.exp(-f.delta_kcal / KT_KCAL_298K);
    });
    var Z = exps.reduce(function(a, b) { return a + b; }, 0) || 1;
    return exps.map(function(e) { return e / Z; });
}

// ---- Sidebar -------------------------------------------------------

function render_sidebar() {
    var state = window.__viewer_state;
    if (!state) return;
    var list = document.getElementById('conformer-list');
    var count = document.getElementById('conformer-count');
    count.textContent = '(' + state.frames.length + ')';
    list.innerHTML = '';

    // Energy-unit toggle's active button reflects state.energy_unit.
    sync_seg_toggle('energy-unit-toggle', state.energy_unit);

    state.sorted_indices.forEach(function(idx) {
        var f = state.frames[idx];
        var card = document.createElement('div');
        card.className = 'conformer-card';
        card.dataset.frameIdx = idx;
        if (idx === state.active_idx) card.classList.add('active');

        var color = gradient_color(f.norm);

        var band = document.createElement('div');
        band.className = 'color-band';
        band.style.background = color;
        card.appendChild(band);

        var body = document.createElement('div');
        body.className = 'card-body';

        var label = document.createElement('div');
        label.className = 'card-label';
        label.textContent = 'Conformer ' + f.index;

        var dE = document.createElement('div');
        dE.className = 'card-dE copy-target';

        // Energy text + tooltip respect state.energy_unit. Double-click
        // copies the displayed numeric value to the clipboard.
        var copy_text = null;
        if (state.energy_unit === 'hartree' && f.energy_hartree !== null) {
            dE.textContent = 'E ' + f.energy_hartree.toFixed(8) + ' Eh';
            copy_text = f.energy_hartree.toFixed(8);
            card.title = '\\u0394E = ' + (f.delta_kcal !== null
                ? f.delta_kcal.toFixed(3) + ' kcal/mol'
                : 'unknown');
        } else if (f.delta_kcal !== null) {
            dE.textContent = '\\u0394E ' + f.delta_kcal.toFixed(2) + ' kcal/mol';
            copy_text = f.delta_kcal.toFixed(4);
            if (f.energy_hartree !== null) {
                card.title = 'E = ' + f.energy_hartree.toFixed(8) + ' Eh';
            }
        } else {
            dE.textContent = '(no energy)';
            card.title = 'No energy parsed from comment line';
        }
        if (copy_text !== null) {
            // Stop single clicks on the energy box from bubbling up
            // to the parent .conformer-card, which would call
            // set_active_conformer → render_sidebar → tear down this
            // dE element. Without this, the dblclick handler that
            // adds .flash-copied was firing on an element that the
            // intervening re-render had already wiped, so the green
            // flash + "Copied!" tooltip never reached the screen.
            // Clipboard write still succeeded (it doesn't care about
            // DOM state), which is why the bug looked like
            // "copy works but no feedback".
            dE.addEventListener('click', function(ev) {
                ev.stopPropagation();
            });
            dE.addEventListener('dblclick', function(ev) {
                ev.stopPropagation();
                ev.preventDefault();
                copy_value(copy_text, dE);
            });
            dE.title = 'double-click to copy';
        }

        // Bar LENGTH = Boltzmann-weight relative to ensemble max
        // (so lowest-E conformer reads as 100% full bar). Bar COLOR
        // stays on the ΔE green→red gradient so the two dimensions
        // stay independently legible.
        var bar_outer = document.createElement('div');
        bar_outer.className = 'bar-outer';
        var bar = document.createElement('div');
        bar.className = 'bar-inner';
        bar.style.width = (f.weight_norm * 100).toFixed(1) + '%';
        bar.style.background = color;
        bar_outer.appendChild(bar);

        body.appendChild(label);
        body.appendChild(dE);
        body.appendChild(bar_outer);
        card.appendChild(body);

        card.addEventListener('click', function() {
            set_active_conformer(parseInt(card.dataset.frameIdx, 10));
        });

        list.appendChild(card);
    });

    // Keep the active card in view if it just changed offscreen (e.g.
    // via keyboard nav with a long list).
    var active_card = list.querySelector('.conformer-card.active');
    if (active_card && active_card.scrollIntoView) {
        active_card.scrollIntoView({ block: 'nearest' });
    }
}

function sync_seg_toggle(toggle_id, active_val) {
    var t = document.getElementById(toggle_id);
    if (!t) return;
    Array.prototype.forEach.call(t.querySelectorAll('button'), function(b) {
        if (b.dataset.val === active_val) {
            b.classList.add('active');
        } else {
            b.classList.remove('active');
        }
    });
}

function gradient_color(t) {
    // t in [0, 1]: 0 = lowest E (green), 1 = highest (red).
    // HSL interpolation through hue, saturation pinned high, lightness
    // mid. Non-perceptually-uniform on purpose (chemist convention).
    var hue = 120 * (1 - t);  // 120° green → 0° red
    return 'hsl(' + hue.toFixed(0) + ', 70%, 50%)';
}

// ---- SMILES block + 2D structure ----------------------------------

function render_smiles_block() {
    var state = window.__viewer_state;
    if (!state) return;
    var block = document.getElementById('smiles-block');
    var val = document.getElementById('smiles-value');
    if (!block || !val) return;
    if (state.smiles) {
        val.textContent = state.smiles;
        block.hidden = false;
    } else {
        val.textContent = '';
        block.hidden = true;
    }
}

function render_2d_structure() {
    var state = window.__viewer_state;
    if (!state || !state.smiles) return;
    var box = document.getElementById('structure-2d');
    var host = document.getElementById('structure-canvas');
    if (!box || !host) return;
    if (!window.__rdkit) {
        // RDKit's WASM init hasn't finished yet. kick_off_rdkit()'s
        // .then will re-call render_2d_structure once it's ready.
        return;
    }
    var mol = null;
    try {
        // PREFERRED: parse the first xyz frame as a mol so the
        // resulting atom indices match the xyz file's atom order
        // exactly (no SMILES→XYZ remapping needed). Some RDKit JS
        // builds don't accept xyz blocks via get_mol — in that case
        // we fall back to the SMILES parse + the ad-hoc atom map
        // (TEST_FIXTURE_ATOM_MAPS). state.uses_xyz_mol records
        // which path we ended up on so align_to_selection knows
        // whether to apply the map.
        mol = try_build_mol_from_xyz_first_frame(state);
        if (mol) {
            state.uses_xyz_mol = true;
        } else {
            state.uses_xyz_mol = false;
            mol = window.__rdkit.get_mol(state.smiles);
        }
        if (!mol || !mol.is_valid()) {
            console.warn('RDKit: invalid SMILES:', state.smiles);
            return;
        }
        // Build the highlights JSON from the current selection.
        // get_svg_with_highlights accepts:
        //   atoms / bonds: indices to highlight
        //   highlightAtomColors / highlightBondColors: maps to RGB
        // Use a single accent color (light blue) so it's visually
        // distinct from RDKit's default heteroatom colors.
        var highlight_atoms = Object.keys(state.selected_atoms || {})
            .map(function(k) { return parseInt(k, 10); });
        var highlight_bonds = derive_highlight_bonds(mol, state.selected_atoms);

        var color = [0.18, 0.52, 0.92];  // #2e85eb-ish (matches the active conformer accent)
        var atomColors = {};
        var bondColors = {};
        highlight_atoms.forEach(function(a) { atomColors[a] = color; });
        highlight_bonds.forEach(function(b) { bondColors[b] = color; });

        var opts = JSON.stringify({
            atoms: highlight_atoms,
            bonds: highlight_bonds,
            highlightAtomColors: atomColors,
            highlightBondColors: bondColors,
            highlightAtomRadii: {},  // default
        });
        var svg = mol.get_svg_with_highlights(opts);
        host.innerHTML = svg;
        // Critical ordering: unhide the inset BEFORE expanding hit
        // areas. expand_svg_hitboxes calls getBBox() on labeled-atom
        // SVG elements to find their actual center, but getBBox()
        // returns zeroes for elements whose ancestor is display:none.
        // If we ran the hitbox pass while box was still hidden, the
        // bond-endpoint fallback would fire and the O hitbox would
        // land at the label edge instead of the label center — visible
        // as "the O is only clickable on its second selection" since
        // subsequent re-renders happen with box visible.
        box.hidden = false;
        // Force a synchronous layout pass so getBBox returns real
        // values rather than zeros for newly-attached elements.
        void box.offsetHeight;
        // Add invisible thicker hit areas for bonds + click circles
        // for unlabeled atoms so the user doesn't have to land on
        // RDKit's thin paths perfectly.
        var svg_root = host.querySelector('svg');
        if (svg_root) expand_svg_hitboxes(svg_root);
        // Re-attach delegated click handler on host (innerHTML wipes
        // any previous listeners on inner SVG nodes; delegation here
        // means we only need one listener regardless of re-renders).
        install_structure_click_handler(host);
        update_selection_toolbar();
    } catch (err) {
        console.warn('RDKit render failed:', err);
    } finally {
        if (mol) {
            try { mol.delete(); } catch (e) { /* ignore */ }
        }
    }
}

function derive_highlight_bonds(mol, selected_atoms) {
    // A bond is highlighted iff both its endpoints are in the
    // selected_atoms set. We get the bond list from RDKit's
    // get_molblock(); each bond line has the two atom indices.
    if (!selected_atoms || Object.keys(selected_atoms).length === 0) {
        return [];
    }
    var bonds = [];
    try {
        var mb = mol.get_molblock();
        // V2000 MOL: counts line has "<natoms> <nbonds> ..." then
        // natoms atom lines, then nbonds bond lines of the form
        // "<a1> <a2> <bond_type> ...". 1-indexed atoms.
        var lines = mb.split('\\n');
        var counts = lines[3].trim().split(/\s+/);
        var n_atoms = parseInt(counts[0], 10);
        var n_bonds = parseInt(counts[1], 10);
        for (var i = 0; i < n_bonds; i++) {
            var line = lines[4 + n_atoms + i];
            // Bond line columns are width-3 each (V2000); fall back
            // to whitespace split if the format is non-standard.
            var a1, a2;
            if (line.length >= 6 && /^\s*\d/.test(line)) {
                a1 = parseInt(line.substring(0, 3), 10) - 1;
                a2 = parseInt(line.substring(3, 6), 10) - 1;
            } else {
                var parts = line.trim().split(/\s+/);
                a1 = parseInt(parts[0], 10) - 1;
                a2 = parseInt(parts[1], 10) - 1;
            }
            if (selected_atoms[a1] && selected_atoms[a2]) bonds.push(i);
        }
    } catch (e) {
        console.warn('derive_highlight_bonds failed:', e);
    }
    return bonds;
}

function install_structure_click_handler(host) {
    if (host.__click_installed) return;
    host.__click_installed = true;
    host.addEventListener('click', function(e) {
        var t = e.target;
        if (!t || !t.getAttribute) return;
        // RDKit's class string can carry several tokens: a bond path
        // has "bond-N atom-A atom-B". Parse all atom-* tokens; if
        // there are two, treat as bond click → toggle both atoms.
        // If exactly one, it's a heteroatom label → toggle that atom.
        var cls = t.getAttribute('class') || '';
        var atom_idx = [];
        cls.split(/\s+/).forEach(function(token) {
            var m = token.match(/^atom-(\d+)$/);
            if (m) atom_idx.push(parseInt(m[1], 10));
        });
        if (atom_idx.length === 0) return;
        e.stopPropagation();
        toggle_atoms(atom_idx);
    });
}

function structure_bg_dblclick(ev) {
    // Double-clicking the inset background EXPANDS to the center
    // overlay (a quick alternative to clicking the chevron). It
    // does NOT collapse — collapse is chevron-only, so a stray
    // double-click on the canvas while picking atoms can't accidentally
    // tear down the expanded view. Atom/bond hits are ignored (they
    // pass through to selection logic).
    var t = ev.target;
    if (t && t.getAttribute) {
        var cls = t.getAttribute('class') || '';
        if (/(?:^|\s)(atom-|bond-)/.test(cls)) return;
    }
    var state = window.__viewer_state;
    if (!state || state.is_expanded) return;
    toggle_structure_expanded(ev);
}

function expand_svg_hitboxes(svg_root) {
    // RDKit's bonds are thin strokes and unlabeled atoms (carbons)
    // have no clickable surface at all. Add invisible "hit area"
    // overlays so the user can click anywhere near a bond/atom.
    //
    // Bonds: clone each bond path with a transparent thicker stroke,
    // insert BEFORE the visible bond in the DOM (so visible art
    // paints over the hitbox), and tag pointer-events on the hit
    // path. Carries the same class string so the click delegation
    // handler treats it identically.
    //
    // Atoms: extract each atom's 2D position from the first bond
    // path endpoint that references it, then add an invisible
    // <circle> of radius 8 (SVG units) tagged with class="atom-N".
    if (!svg_root || svg_root.__hitboxes_added) return;
    svg_root.__hitboxes_added = true;

    var ns = 'http://www.w3.org/2000/svg';
    var bonds = svg_root.querySelectorAll('[class*="bond-"]');

    var atom_pos = {};  // atom_idx → [x, y]
    bonds.forEach(function(b) {
        var cls = b.getAttribute('class') || '';
        var ais = [];
        cls.split(/\s+/).forEach(function(tok) {
            var m = tok.match(/^atom-(\d+)$/);
            if (m) ais.push(parseInt(m[1], 10));
        });
        var d = b.getAttribute('d') || '';
        // Pull numeric tokens from the d attr; bond paths typically
        // start with "M x1,y1 L x2,y2" but double bonds add a second
        // segment. First two pairs are the bond endpoints.
        var nums = d.match(/-?\\d+(?:\\.\\d+)?/g) || [];
        if (ais.length === 2 && nums.length >= 4) {
            if (atom_pos[ais[0]] === undefined) {
                atom_pos[ais[0]] = [parseFloat(nums[0]), parseFloat(nums[1])];
            }
            if (atom_pos[ais[1]] === undefined) {
                atom_pos[ais[1]] = [parseFloat(nums[2]), parseFloat(nums[3])];
            }
        }
        // Insert invisible thick-stroke clone behind the visible bond
        var hit = b.cloneNode(false);
        hit.setAttribute('stroke', 'transparent');
        hit.setAttribute('stroke-width', '14');
        hit.setAttribute('fill', 'none');
        hit.setAttribute('pointer-events', 'stroke');
        hit.removeAttribute('style');
        b.parentNode.insertBefore(hit, b);
    });

    // For atoms RDKit renders with a visible label (heteroatoms like
    // O, N, S, …), bond paths are TRUNCATED to the label's bounding
    // box, so the bond-endpoint positions above are offset from the
    // true atom centers. Override using the actual rendered bbox of
    // each labeled atom element — this works regardless of how
    // RDKit nests text (sometimes <text class="atom-N">, sometimes
    // <g class="atom-N"><text>...</text></g>, sometimes split across
    // <tspan>s for subscripts).
    var atom_classed = svg_root.querySelectorAll('[class*="atom-"]');
    atom_classed.forEach(function(el) {
        var cls = el.getAttribute('class') || '';
        if (/(?:^|\\s)(atom-hitbox|bond-)/.test(cls)) return;
        var atom_tokens = cls.match(/atom-\\d+/g) || [];
        // Bonds have two atom-* tokens (atom-A atom-B); atoms have one.
        if (atom_tokens.length !== 1) return;
        var idx = parseInt(atom_tokens[0].replace('atom-', ''), 10);
        var bbox;
        try {
            bbox = el.getBBox();
        } catch (e) {
            return;
        }
        if (bbox && bbox.width > 0 && bbox.height > 0) {
            atom_pos[idx] = [bbox.x + bbox.width / 2,
                             bbox.y + bbox.height / 2];
        }
    });

    // Add invisible click circles per atom. Radius 11 (up from 9) so
    // labeled-atom glyphs are fully covered by the hit area.
    Object.keys(atom_pos).forEach(function(idx) {
        var p = atom_pos[idx];
        var c = document.createElementNS(ns, 'circle');
        c.setAttribute('cx', p[0]);
        c.setAttribute('cy', p[1]);
        c.setAttribute('r', '11');
        c.setAttribute('fill', 'transparent');
        c.setAttribute('pointer-events', 'all');
        c.setAttribute('class', 'atom-' + idx + ' atom-hitbox');
        svg_root.appendChild(c);
    });
}

function toggle_atoms(indices) {
    var state = window.__viewer_state;
    if (!state) return;
    // If every index in `indices` is already selected, toggle them
    // ALL off (so clicking a fully-selected bond clears it).
    // Otherwise, add the ones that aren't.
    var all_set = indices.every(function(i) { return state.selected_atoms[i]; });
    indices.forEach(function(i) {
        if (all_set) {
            delete state.selected_atoms[i];
        } else {
            state.selected_atoms[i] = true;
        }
    });
    state.selection_count = Object.keys(state.selected_atoms).length;
    render_2d_structure();
}

function update_selection_toolbar() {
    var state = window.__viewer_state;
    if (!state) return;
    var count_el = document.getElementById('structure-selection-count');
    var align_btn = document.querySelector('.sb-action.align');
    var reset_btn = document.querySelector('.sb-action.reset');
    var n = state.selection_count || 0;
    if (count_el) {
        count_el.textContent = n === 0
            ? 'No atoms selected'
            : n + ' atom' + (n === 1 ? '' : 's') + ' selected';
    }
    if (align_btn) {
        // Three alignment modes by selection size:
        //   n=1 — pure translation (move the atom to match reference)
        //   n=2 — translation of midpoint + minimal axis rotation of bond
        //   n≥3 — full Kabsch
        align_btn.disabled = n < 1;
        align_btn.title = (n < 1)
            ? 'Select at least one atom to compute alignment'
            : ('Apply ' +
               (n === 1 ? 'translation' :
                n === 2 ? 'translation + axis rotation' :
                'Kabsch') + ' alignment to all conformers');
    }
    if (reset_btn) {
        reset_btn.disabled = (n === 0) && !state.is_aligned;
    }
}

// ---- Inset expand/compact toggle -----------------------------------

function toggle_structure_expanded(ev) {
    if (ev) {
        ev.preventDefault();
        ev.stopPropagation();
    }
    var state = window.__viewer_state;
    var box = document.getElementById('structure-2d');
    if (!state || !box) return;
    state.is_expanded = !state.is_expanded;
    if (state.is_expanded) {
        box.classList.add('expanded');
    } else {
        box.classList.remove('expanded');
    }
    // Chevron tooltip reflects what the next click will do.
    var btn = document.getElementById('structure-expand-btn');
    if (btn) {
        btn.title = state.is_expanded ? 'Collapse' : 'Expand to center';
    }
}

// ---- Kabsch alignment math ----------------------------------------

function ensure_coords_cache() {
    // First call: snapshot every atom's xyz in every model. Used by
    // align_to_selection (as the source) and reset_alignment (as the
    // restore target). Cheap memory-wise — N_frames * N_atoms * 3
    // doubles, typically << 1 MB.
    var state = window.__viewer_state;
    if (!state || state.original_coords) return;
    var cached = [];
    state.frames.forEach(function(_, frame_idx) {
        var model = state.viewer.getModel(frame_idx);
        var atoms = model.selectedAtoms({});
        cached.push(atoms.map(function(a) { return [a.x, a.y, a.z]; }));
    });
    state.original_coords = cached;
}

function jacobi_eigen_3x3_sym(A_in) {
    // Eigendecomposition of a 3x3 SYMMETRIC matrix via Jacobi
    // rotations. Returns {values: [v1, v2, v3], vectors: V} where
    // V[i][j] = the i-th component of the j-th eigenvector
    // (columns are eigenvectors).
    var A = A_in.map(function(row) { return row.slice(); });
    var V = [[1,0,0],[0,1,0],[0,0,1]];
    for (var iter = 0; iter < 60; iter++) {
        // Find off-diagonal element with largest absolute value.
        var p = 0, q = 1, max_off = Math.abs(A[0][1]);
        if (Math.abs(A[0][2]) > max_off) { p = 0; q = 2; max_off = Math.abs(A[0][2]); }
        if (Math.abs(A[1][2]) > max_off) { p = 1; q = 2; max_off = Math.abs(A[1][2]); }
        if (max_off < 1e-12) break;
        // Compute rotation parameters for Givens.
        var theta = (A[q][q] - A[p][p]) / (2 * A[p][q]);
        var t;
        if (Math.abs(theta) > 1e6) {
            t = 0.5 / theta;
        } else {
            t = (theta >= 0 ? 1 : -1)
                / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        }
        var c = 1 / Math.sqrt(t * t + 1);
        var s = c * t;
        var Apq = A[p][q];
        A[p][p] = A[p][p] - t * Apq;
        A[q][q] = A[q][q] + t * Apq;
        A[p][q] = 0;
        A[q][p] = 0;
        for (var r = 0; r < 3; r++) {
            if (r !== p && r !== q) {
                var Arp = A[r][p], Arq = A[r][q];
                A[r][p] = c * Arp - s * Arq;
                A[p][r] = A[r][p];
                A[r][q] = s * Arp + c * Arq;
                A[q][r] = A[r][q];
            }
        }
        for (var r = 0; r < 3; r++) {
            var Vrp = V[r][p], Vrq = V[r][q];
            V[r][p] = c * Vrp - s * Vrq;
            V[r][q] = s * Vrp + c * Vrq;
        }
    }
    return { values: [A[0][0], A[1][1], A[2][2]], vectors: V };
}

function kabsch_rotation(P_centered, Q_centered) {
    // Standard Kabsch derivation:
    //   H = Qᵀ · P  (3×3 cross-covariance)
    //   SVD: H = U Σ Vᵀ
    //   R_optimal = V · diag(1, 1, sign(det(V Uᵀ))) · Uᵀ
    // Returns R such that R · Q_centered[i] ≈ P_centered[i] for all i.
    //
    // We compute V from eigendecomp of HᵀH, then derive U = H · V · Σ⁻¹.
    // Computing U and V independently via two eigendecomps doesn't
    // work because the column orderings + signs come out arbitrary.
    var n = P_centered.length;
    if (n === 0) return [[1,0,0],[0,1,0],[0,0,1]];

    // H[a][b] = (Qᵀ P)[a][b] = Σ_i Q[i][a] · P[i][b]
    var H = [[0,0,0],[0,0,0],[0,0,0]];
    for (var i = 0; i < n; i++) {
        for (var a = 0; a < 3; a++) {
            for (var b = 0; b < 3; b++) {
                H[a][b] += Q_centered[i][a] * P_centered[i][b];
            }
        }
    }

    // HᵀH — 3×3 symmetric. Eigenvectors → V; eigenvalues → σ².
    var HtH = [[0,0,0],[0,0,0],[0,0,0]];
    for (var a = 0; a < 3; a++) {
        for (var b = 0; b < 3; b++) {
            for (var k = 0; k < 3; k++) {
                HtH[a][b] += H[k][a] * H[k][b];
            }
        }
    }
    var eV = jacobi_eigen_3x3_sym(HtH);
    var V = eV.vectors;  // V[i][j] = i-th component of j-th eigenvector
    var sigma = eV.values.map(function(v) { return Math.sqrt(Math.max(v, 0)); });

    // U columns: U_j = (H · V_j) / σ_j. Keeps sign coherent with V.
    // CRITICAL: when σ_j is near zero (rank-deficient H, common when
    // the selected substructure is nearly coplanar / colinear), the
    // division yields garbage and the resulting R is a projection
    // (det near 0), which manifests as the molecule visibly
    // flattening onto a plane. We detect this case and complete the
    // missing U columns via cross products of the good ones, which
    // keeps U orthonormal and R a proper rotation.
    var U = [[0,0,0],[0,0,0],[0,0,0]];
    var sigma_max = Math.max(sigma[0], sigma[1], sigma[2]);
    var deg_threshold = sigma_max * 1e-5 + 1e-10;
    var good_cols = [];
    for (var j = 0; j < 3; j++) {
        if (sigma[j] > deg_threshold) {
            var HVj0 = H[0][0]*V[0][j] + H[0][1]*V[1][j] + H[0][2]*V[2][j];
            var HVj1 = H[1][0]*V[0][j] + H[1][1]*V[1][j] + H[1][2]*V[2][j];
            var HVj2 = H[2][0]*V[0][j] + H[2][1]*V[1][j] + H[2][2]*V[2][j];
            U[0][j] = HVj0 / sigma[j];
            U[1][j] = HVj1 / sigma[j];
            U[2][j] = HVj2 / sigma[j];
            good_cols.push(j);
        }
    }
    complete_U_orthonormal(U, good_cols);

    // R = V · Uᵀ.  Uᵀ[i][j] = U[j][i] → (V·Uᵀ)[i][j] = Σ_k V[i][k] · U[j][k].
    function matmul_VUt(V_, U_) {
        var M = [[0,0,0],[0,0,0],[0,0,0]];
        for (var i = 0; i < 3; i++) {
            for (var j = 0; j < 3; j++) {
                for (var k = 0; k < 3; k++) {
                    M[i][j] += V_[i][k] * U_[j][k];
                }
            }
        }
        return M;
    }
    function det3(M) {
        return M[0][0]*(M[1][1]*M[2][2] - M[1][2]*M[2][1])
             - M[0][1]*(M[1][0]*M[2][2] - M[1][2]*M[2][0])
             + M[0][2]*(M[1][0]*M[2][1] - M[1][1]*M[2][0]);
    }
    var R = matmul_VUt(V, U);
    if (det3(R) < 0) {
        // det(R) = ±1; if negative we have a reflection. Restore a
        // proper rotation by flipping the V column for the smallest σ.
        var idx_min = 0;
        if (sigma[1] < sigma[idx_min]) idx_min = 1;
        if (sigma[2] < sigma[idx_min]) idx_min = 2;
        V[0][idx_min] = -V[0][idx_min];
        V[1][idx_min] = -V[1][idx_min];
        V[2][idx_min] = -V[2][idx_min];
        R = matmul_VUt(V, U);
    }
    return R;
}

function complete_U_orthonormal(U, good_cols) {
    // Given the columns of U that we computed from H · V / σ (the
    // "good" ones, where σ was non-degenerate), fill in the missing
    // columns so U is a proper orthonormal 3×3 matrix.
    //   3 good → already orthonormal (by construction); ensure unit.
    //   2 good → fill the third via cross product of the two.
    //   1 good → pick a perpendicular axis arbitrarily, complete via cross.
    //   0 good → identity (no rotational info; only translation makes sense).
    function colnorm(j) {
        return Math.sqrt(U[0][j]*U[0][j] + U[1][j]*U[1][j] + U[2][j]*U[2][j]);
    }
    function normalize(j) {
        var n = colnorm(j);
        if (n > 1e-12) {
            U[0][j] /= n; U[1][j] /= n; U[2][j] /= n;
        }
    }
    function cross_into(j, a, b) {
        U[0][j] = U[1][a]*U[2][b] - U[2][a]*U[1][b];
        U[1][j] = U[2][a]*U[0][b] - U[0][a]*U[2][b];
        U[2][j] = U[0][a]*U[1][b] - U[1][a]*U[0][b];
    }
    good_cols.forEach(normalize);
    if (good_cols.length === 3) return;
    if (good_cols.length === 2) {
        var a = good_cols[0], b = good_cols[1];
        var missing = [0,1,2].filter(function(x) { return x !== a && x !== b; })[0];
        cross_into(missing, a, b);
        normalize(missing);
        return;
    }
    if (good_cols.length === 1) {
        var a = good_cols[0];
        // Find a unit vector perpendicular to U[:,a]
        var ca = [U[0][a], U[1][a], U[2][a]];
        var seed = Math.abs(ca[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
        var dot = seed[0]*ca[0] + seed[1]*ca[1] + seed[2]*ca[2];
        var perp = [seed[0]-dot*ca[0], seed[1]-dot*ca[1], seed[2]-dot*ca[2]];
        var pn = Math.sqrt(perp[0]*perp[0] + perp[1]*perp[1] + perp[2]*perp[2]);
        perp = [perp[0]/pn, perp[1]/pn, perp[2]/pn];
        var others = [0,1,2].filter(function(x) { return x !== a; });
        U[0][others[0]] = perp[0]; U[1][others[0]] = perp[1]; U[2][others[0]] = perp[2];
        cross_into(others[1], a, others[0]);
        normalize(others[1]);
        return;
    }
    // good_cols.length === 0 → identity
    U[0][0] = 1; U[1][0] = 0; U[2][0] = 0;
    U[0][1] = 0; U[1][1] = 1; U[2][1] = 0;
    U[0][2] = 0; U[1][2] = 0; U[2][2] = 1;
}

function compute_kabsch_transform(P, Q) {
    // P = N×3 reference coords, Q = N×3 target coords (to be aligned).
    // Returns {R, t} such that R·Q[i] + t ≈ P[i] for all i in P.
    var n = P.length;
    var pc = [0,0,0], qc = [0,0,0];
    for (var i = 0; i < n; i++) {
        for (var k = 0; k < 3; k++) {
            pc[k] += P[i][k];
            qc[k] += Q[i][k];
        }
    }
    for (var k = 0; k < 3; k++) { pc[k] /= n; qc[k] /= n; }
    var Pc = P.map(function(p) { return [p[0]-pc[0], p[1]-pc[1], p[2]-pc[2]]; });
    var Qc = Q.map(function(q) { return [q[0]-qc[0], q[1]-qc[1], q[2]-qc[2]]; });
    var R = kabsch_rotation(Pc, Qc);
    var Rqc = [
        R[0][0]*qc[0] + R[0][1]*qc[1] + R[0][2]*qc[2],
        R[1][0]*qc[0] + R[1][1]*qc[1] + R[1][2]*qc[2],
        R[2][0]*qc[0] + R[2][1]*qc[1] + R[2][2]*qc[2],
    ];
    return {
        R: R,
        t: [pc[0] - Rqc[0], pc[1] - Rqc[1], pc[2] - Rqc[2]],
    };
}

function apply_transform_inplace(model, R, t) {
    var atoms = model.selectedAtoms({});
    atoms.forEach(function(a) {
        var x = a.x, y = a.y, z = a.z;
        a.x = R[0][0]*x + R[0][1]*y + R[0][2]*z + t[0];
        a.y = R[1][0]*x + R[1][1]*y + R[1][2]*z + t[1];
        a.z = R[2][0]*x + R[2][1]*y + R[2][2]*z + t[2];
    });
}

function set_model_coords(model, coords) {
    // Restore atom positions from a snapshotted N×3 array.
    var atoms = model.selectedAtoms({});
    atoms.forEach(function(a, i) {
        if (!coords[i]) return;
        a.x = coords[i][0];
        a.y = coords[i][1];
        a.z = coords[i][2];
    });
}

// ---- Align / Reset actions ----------------------------------------

function compute_translation_transform(P_pt, Q_pt) {
    // Single-atom alignment: R = identity, t = P - Q. Applied to all
    // atoms of the target, this shifts the molecule so Q[0] lands on
    // P[0] with orientation unchanged.
    return {
        R: [[1,0,0],[0,1,0],[0,0,1]],
        t: [P_pt[0] - Q_pt[0], P_pt[1] - Q_pt[1], P_pt[2] - Q_pt[2]],
    };
}

function compute_two_atom_transform(P, Q) {
    // Two-atom alignment: translate Q's midpoint to P's midpoint,
    // then apply the *minimal* rotation that aligns the Q-bond
    // vector with the P-bond vector. Endpoints don't get flipped —
    // P[0] ↔ Q[0] and P[1] ↔ Q[1] specifically.
    var p_mid = [(P[0][0]+P[1][0])/2, (P[0][1]+P[1][1])/2, (P[0][2]+P[1][2])/2];
    var q_mid = [(Q[0][0]+Q[1][0])/2, (Q[0][1]+Q[1][1])/2, (Q[0][2]+Q[1][2])/2];
    var p_vec = [P[1][0]-P[0][0], P[1][1]-P[0][1], P[1][2]-P[0][2]];
    var q_vec = [Q[1][0]-Q[0][0], Q[1][1]-Q[0][1], Q[1][2]-Q[0][2]];
    var p_len = Math.sqrt(p_vec[0]*p_vec[0] + p_vec[1]*p_vec[1] + p_vec[2]*p_vec[2]);
    var q_len = Math.sqrt(q_vec[0]*q_vec[0] + q_vec[1]*q_vec[1] + q_vec[2]*q_vec[2]);
    if (p_len < 1e-10 || q_len < 1e-10) {
        // Picks coincide; degrade to translation only.
        return {
            R: [[1,0,0],[0,1,0],[0,0,1]],
            t: [p_mid[0]-q_mid[0], p_mid[1]-q_mid[1], p_mid[2]-q_mid[2]],
        };
    }
    var pu = [p_vec[0]/p_len, p_vec[1]/p_len, p_vec[2]/p_len];
    var qu = [q_vec[0]/q_len, q_vec[1]/q_len, q_vec[2]/q_len];
    // Rotation taking qu → pu via Rodrigues' formula.
    // axis = qu × pu (rotate FROM qu TO pu so the cross is taken in that order),
    // sin(θ) = |axis|, cos(θ) = qu · pu.
    var axis = [
        qu[1]*pu[2] - qu[2]*pu[1],
        qu[2]*pu[0] - qu[0]*pu[2],
        qu[0]*pu[1] - qu[1]*pu[0],
    ];
    var sin_t = Math.sqrt(axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2]);
    var cos_t = qu[0]*pu[0] + qu[1]*pu[1] + qu[2]*pu[2];
    var R;
    if (sin_t < 1e-10) {
        // qu and pu are (anti-)parallel.
        if (cos_t > 0) {
            R = [[1,0,0],[0,1,0],[0,0,1]];
        } else {
            // 180° rotation about any axis perpendicular to qu.
            var seed = Math.abs(qu[0]) < 0.9 ? [1,0,0] : [0,1,0];
            var d = seed[0]*qu[0] + seed[1]*qu[1] + seed[2]*qu[2];
            var perp = [seed[0]-d*qu[0], seed[1]-d*qu[1], seed[2]-d*qu[2]];
            var pn = Math.sqrt(perp[0]*perp[0]+perp[1]*perp[1]+perp[2]*perp[2]);
            var u = [perp[0]/pn, perp[1]/pn, perp[2]/pn];
            R = [
                [2*u[0]*u[0]-1, 2*u[0]*u[1],   2*u[0]*u[2]  ],
                [2*u[1]*u[0],   2*u[1]*u[1]-1, 2*u[1]*u[2]  ],
                [2*u[2]*u[0],   2*u[2]*u[1],   2*u[2]*u[2]-1],
            ];
        }
    } else {
        var k = [axis[0]/sin_t, axis[1]/sin_t, axis[2]/sin_t];
        var C = 1 - cos_t;
        R = [
            [cos_t + k[0]*k[0]*C,    k[0]*k[1]*C - k[2]*sin_t, k[0]*k[2]*C + k[1]*sin_t],
            [k[1]*k[0]*C + k[2]*sin_t, cos_t + k[1]*k[1]*C,    k[1]*k[2]*C - k[0]*sin_t],
            [k[2]*k[0]*C - k[1]*sin_t, k[2]*k[1]*C + k[0]*sin_t, cos_t + k[2]*k[2]*C   ],
        ];
    }
    var Rq = [
        R[0][0]*q_mid[0] + R[0][1]*q_mid[1] + R[0][2]*q_mid[2],
        R[1][0]*q_mid[0] + R[1][1]*q_mid[1] + R[1][2]*q_mid[2],
        R[2][0]*q_mid[0] + R[2][1]*q_mid[1] + R[2][2]*q_mid[2],
    ];
    return {
        R: R,
        t: [p_mid[0]-Rq[0], p_mid[1]-Rq[1], p_mid[2]-Rq[2]],
    };
}

function align_to_selection() {
    var state = window.__viewer_state;
    if (!state) return;
    var n = state.selection_count || 0;
    if (n < 1) return;
    ensure_coords_cache();

    // SMILES-space atom indices (from the 2D RDKit drawing).
    var smiles_indices = Object.keys(state.selected_atoms)
        .map(function(k) { return parseInt(k, 10); })
        .filter(function(i) { return Number.isFinite(i); });

    // Translate the 2D-drawing atom indices to xyz-file atom indices.
    // Three cases:
    //   * uses_xyz_mol === true: drawing was made from the xyz, so
    //     atom-N in the SVG IS xyz atom N. Identity map.
    //   * SMILES-based drawing with a registered ad-hoc map: apply
    //     that map (per the test fixture).
    //   * SMILES-based drawing with no map: assume identity (correct
    //     for workflow-produced ensembles where smiles_to_3d
    //     preserves RDKit's atom order through to xyz).
    var atom_map = state.uses_xyz_mol
        ? null
        : get_atom_map_for_smiles(state.smiles);
    var xyz_indices = smiles_indices.map(function(i) {
        return atom_map ? atom_map[i] : i;
    });
    if (xyz_indices.some(function(i) { return i === undefined || i === null; })) {
        console.warn('Align: atom map missing some indices', smiles_indices);
        return;
    }

    var ref_coords = state.original_coords[state.active_idx];
    var P = xyz_indices.map(function(i) { return ref_coords[i]; });
    if (P.some(function(p) { return !p; })) {
        console.warn('Align: some xyz indices out of range for active conformer');
        return;
    }

    var mode = (n === 1) ? 'translation'
             : (n === 2) ? 'two-atom'
             : 'kabsch';
    console.log('Align (' + mode + ') with smiles idx ' +
                JSON.stringify(smiles_indices) +
                ' → xyz idx ' + JSON.stringify(xyz_indices));

    state.frames.forEach(function(_, frame_idx) {
        var model = state.viewer.getModel(frame_idx);
        // First, restore the model to its original coords so we're
        // computing the transform from a known baseline.
        set_model_coords(model, state.original_coords[frame_idx]);
        if (frame_idx === state.active_idx) return;
        var Q = xyz_indices.map(function(i) {
            return state.original_coords[frame_idx][i];
        });
        if (Q.some(function(q) { return !q; })) return;
        var tr;
        if (mode === 'translation')  tr = compute_translation_transform(P[0], Q[0]);
        else if (mode === 'two-atom') tr = compute_two_atom_transform(P, Q);
        else                          tr = compute_kabsch_transform(P, Q);
        apply_transform_inplace(model, tr.R, tr.t);
    });

    state.is_aligned = true;
    // repaint_viewer() forces 3Dmol to re-derive bond geometry from
    // the new atom positions; calling viewer.render() alone left
    // bonds drawn at old positions until you toggled conformer.
    repaint_viewer();
    update_selection_toolbar();
}

function reset_alignment() {
    var state = window.__viewer_state;
    if (!state) return;
    // Restore original coordinates if we have them.
    if (state.original_coords) {
        state.frames.forEach(function(_, frame_idx) {
            set_model_coords(
                state.viewer.getModel(frame_idx),
                state.original_coords[frame_idx]
            );
        });
        // Force 3Dmol to re-derive bond geometry from the restored
        // positions instead of just redrawing atoms in place.
        repaint_viewer();
    }
    // Clear selection.
    state.selected_atoms = {};
    state.selection_count = 0;
    state.is_aligned = false;
    render_2d_structure();
}

// ---- Sparkline -----------------------------------------------------

function render_sparkline() {
    var state = window.__viewer_state;
    if (!state) return;
    var svg = document.getElementById('sparkline');
    var range_el = document.getElementById('sparkline-range');
    if (!svg) return;
    while (svg.firstChild) svg.removeChild(svg.firstChild);

    sync_seg_toggle('chart-axis-toggle', state.chart_y_axis);

    var data = state.sorted_indices.map(function(frame_idx) {
        var f = state.frames[frame_idx];
        var v;
        if (state.chart_y_axis === 'boltzmann') {
            v = (state.boltzmann_weights[frame_idx] || 0) * 100;
        } else {
            v = (f.delta_kcal === null ? 0 : f.delta_kcal);
        }
        return { frame_idx: frame_idx, value: v };
    });
    if (data.length === 0) return;

    var values = data.map(function(d) { return d.value; });
    var vmin = Math.min.apply(null, values);
    var vmax = Math.max.apply(null, values);
    var vrange = (vmax - vmin) || 1;

    // viewBox is 280×80 (fixed in HTML); SVG preserveAspectRatio=none
    // stretches it to whatever width the sidebar gives us.
    var W = 280, H = 80;
    var PAD_X = 8, PAD_Y = 6;
    var inner_w = W - 2 * PAD_X;
    var inner_h = H - 2 * PAD_Y;
    var n = data.length;

    function x_of(i) {
        return n > 1
            ? PAD_X + (i / (n - 1)) * inner_w
            : PAD_X + inner_w / 2;
    }
    function y_of(v) {
        return PAD_Y + (1 - (v - vmin) / vrange) * inner_h;
    }

    // Connecting polyline (thin grey).
    var pts = data.map(function(d, i) {
        return x_of(i) + ',' + y_of(d.value);
    }).join(' ');
    svg.appendChild(svg_elem('polyline', {
        points: pts,
        fill: 'none',
        stroke: '#bcbcc4',
        'stroke-width': '1',
        'stroke-linejoin': 'round',
    }));

    // Per-point circles. Active one is highlighted; others greyscale.
    data.forEach(function(d, i) {
        var is_active = d.frame_idx === state.active_idx;
        var c = svg_elem('circle', {
            cx: x_of(i).toFixed(2),
            cy: y_of(d.value).toFixed(2),
            r: is_active ? 4 : 2.5,
            fill: is_active ? '#2152a4' : '#9a9aa3',
            stroke: '#fff',
            'stroke-width': '1',
        });
        c.style.cursor = 'pointer';
        c.dataset.frameIdx = d.frame_idx;
        var tooltip = 'Conformer ' + state.frames[d.frame_idx].index
            + ': ' + (state.chart_y_axis === 'boltzmann'
                ? d.value.toFixed(1) + '%'
                : '\\u0394E ' + d.value.toFixed(3) + ' kcal/mol');
        c.appendChild(svg_elem('title', {})).textContent = tooltip;
        // (textContent assignment on the just-appended <title> works
        // in all browsers; the appendChild returns the inserted node.)
        c.addEventListener('click', function() {
            set_active_conformer(d.frame_idx);
        });
        svg.appendChild(c);
    });

    if (range_el) {
        if (state.chart_y_axis === 'boltzmann') {
            range_el.textContent = vmin.toFixed(1) + '% – ' + vmax.toFixed(1) + '%';
        } else {
            range_el.textContent = '0 – ' + vmax.toFixed(2) + ' kcal/mol';
        }
    }
}

// ---- Render passes -------------------------------------------------

function set_active_conformer(idx) {
    var state = window.__viewer_state;
    if (!state) return;
    if (idx < 0 || idx >= state.frames.length) return;
    state.active_idx = idx;
    render_sidebar();
    render_sparkline();
    repaint_viewer();
}

function set_view_mode(mode) {
    var state = window.__viewer_state;
    if (!state) return;
    state.mode = mode;
    repaint_viewer();
}

function apply_style(style_name) {
    var state = window.__viewer_state;
    if (!state) return;
    state.style = style_name;
    repaint_viewer();
}

function set_energy_unit(unit) {
    var state = window.__viewer_state;
    if (!state || (unit !== 'kcal' && unit !== 'hartree')) return;
    state.energy_unit = unit;
    render_sidebar();
}

function set_chart_axis(axis) {
    var state = window.__viewer_state;
    if (!state || (axis !== 'energy' && axis !== 'boltzmann')) return;
    state.chart_y_axis = axis;
    render_sparkline();
}

// 3Dmol hover/click callbacks — module-level so repaint_viewer can
// re-register them per-model without rebuilding closures every frame.

function _viewer_hover_cb(atom /*, viewer_, ev, container*/) {
    var st = window.__viewer_state;
    if (!st || !atom) return;
    // Should already be the active model (the selection scoping below
    // limits hover registration to it), but keep the check as a
    // belt-and-suspenders against stale callbacks.
    if (atom.model !== st.active_idx) return;
    var info_el = document.getElementById('atom-hover-info');
    if (!info_el) return;
    var serial = (atom.serial != null) ? atom.serial : '';
    info_el.textContent = (atom.elem || '?') + serial;
    info_el.classList.add('visible');
}

function _viewer_unhover_cb(/*atom, viewer_*/) {
    var info_el = document.getElementById('atom-hover-info');
    if (info_el) info_el.classList.remove('visible');
}

function _viewer_click_cb(atom /*, viewer_, ev, container*/) {
    var st = window.__viewer_state;
    if (!st || !atom) return;
    if (atom.model !== st.active_idx) return;
    var pin_el = document.getElementById('atom-pinned-label');
    if (!pin_el) return;
    var serial = (atom.serial != null) ? atom.serial : '';
    var label = (atom.elem || '?') + serial;
    if (pin_el.textContent === label &&
        pin_el.classList.contains('visible')) {
        pin_el.classList.remove('visible');
        pin_el.textContent = '';
    } else {
        pin_el.textContent = label;
        pin_el.classList.add('visible');
    }
}

function repaint_viewer() {
    var state = window.__viewer_state;
    if (!state) return;
    var v = state.viewer;
    var active_style = build_style_spec(state.style, /*active=*/true);
    var overlay_style = build_style_spec(state.style, /*active=*/false);

    state.frames.forEach(function(_, i) {
        if (i === state.active_idx) {
            v.setStyle({ model: i }, active_style);
            // Only the active conformer's atoms accept hover/click —
            // otherwise the inactive conformers' line geometry (the
            // grey overlay sticks) eats picking events the user
            // expects to land on the visible active atom underneath.
            v.setHoverable({ model: i }, true,
                           _viewer_hover_cb, _viewer_unhover_cb);
            v.setClickable({ model: i }, true, _viewer_click_cb);
        } else {
            if (state.mode === 'overlay') {
                v.setStyle({ model: i }, overlay_style);
            } else {
                // single-conformer mode hides non-active models.
                v.setStyle({ model: i }, {});
            }
            v.setHoverable({ model: i }, false);
            v.setClickable({ model: i }, false);
        }
    });
    v.render();
}

function build_style_spec(style_name, is_active) {
    if (is_active) {
        // Active conformer: full-color, normal radii.
        switch (style_name) {
            case 'stick':     return { stick: { radius: 0.15 } };
            case 'sphere':    return { sphere: { scale: 1.0 } };
            case 'line':      return { line: { linewidth: 2.0 } };
            case 'ballstick':
            default:          return {
                stick: { radius: 0.15 },
                sphere: { scale: 0.25 }
            };
        }
    } else {
        // Overlay (inactive) conformers: thin grey lines so they
        // suggest the population without competing for attention.
        // 3Dmol's color: 'grey' + low linewidth.
        return { line: { linewidth: 1.0, color: '#bbbbbb' } };
    }
}

function reset_view() {
    var state = window.__viewer_state;
    if (!state) return;
    state.viewer.zoomTo();
    state.viewer.render();
}
""".replace("__OPAAT_SERVICE__", OPAAT_SERVICE).replace(
    "__OPAAT_RESOURCE_PREFIX__", OPAAT_RESOURCE_PREFIX
)


STYLES_CSS = """\
/* ensemble_viewer — viewer + sidebar layout. */

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
    gap: 8px;
    font-size: 14px;
    height: 36px;
    box-sizing: border-box;
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

#content {
    display: flex;
    flex-direction: row;
    height: calc(100vh - 53px);  /* 36px toolbar + ~17px border */
    box-sizing: border-box;
}

#viewer {
    position: relative;
    flex: 1 1 auto;
    height: 100%;
    min-width: 0;
}

/* Atom-hover readout pinned to the bottom-right of the 3D viewer.
   Empty / invisible by default; ``.visible`` is toggled by the
   3Dmol setHoverable callback when the cursor is over an atom of
   the active conformer. */
#atom-hover-info {
    position: absolute;
    bottom: 12px;
    right: 12px;
    background: rgba(255, 255, 255, 0.94);
    border: 1px solid #d8d8dc;
    border-radius: 4px;
    padding: 3px 9px;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 12px;
    color: #222;
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.12s ease;
    z-index: 40;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
    min-width: 36px;
    text-align: center;
}
#atom-hover-info.visible {
    opacity: 1;
}

/* Click-to-pin atom label: bottom-center, accent-colored to feel
   "selected". Empty / invisible until an atom in the active
   conformer is clicked. Clicking the same atom again clears it. */
#atom-pinned-label {
    position: absolute;
    bottom: 12px;
    left: 50%;
    transform: translateX(-50%);
    background: rgba(33, 82, 164, 0.92);
    color: #fff;
    border-radius: 4px;
    padding: 4px 14px;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.02em;
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.15s ease;
    z-index: 40;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.18);
}
#atom-pinned-label.visible {
    opacity: 1;
}

#sidebar {
    width: 320px;
    flex: 0 0 320px;
    border-left: 1px solid #d8d8dc;
    background: #fafafb;
    overflow-y: auto;
    overflow-x: hidden;
    display: flex;
    flex-direction: column;
}

/* SMILES block: free-floating at the top of the sidebar, scrolls
   with the content (not sticky — operator can scroll past it once
   they're focused on conformers). */
#smiles-block {
    padding: 10px 12px 8px;
    border-bottom: 1px solid #e4e4e8;
    background: #fff;
    font-size: 12px;
}
#smiles-block[hidden] {
    display: none;
}
.sb-label {
    color: #555;
    font-size: 11px;
    margin-bottom: 3px;
    letter-spacing: 0.02em;
    text-transform: uppercase;
}
.sb-hint {
    text-transform: none;
    letter-spacing: 0;
    color: #888;
    font-style: italic;
    font-size: 10px;
    margin-left: 4px;
}
#smiles-value {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 12px;
    color: #222;
    word-break: break-all;
    cursor: copy;
    padding: 4px 6px;
    background: #f6f6f9;
    border-radius: 3px;
    border: 1px solid #e8e8ec;
    user-select: text;
    position: relative;
}

/* Sticky sidebar sections. The header and the sparkline both pin
   to the top so they stay visible while the conformer list scrolls.
   They stack via incrementing ``top`` offsets. */
.sidebar-section {
    background: #fafafb;
}
.sticky-top {
    position: sticky;
    top: 0;
    z-index: 3;
    border-bottom: 1px solid #e4e4e8;
}
.sticky-below-header {
    /* The header above is roughly 36px tall (single line of content
       with padding). 38px gives a hair of margin so a 1px sub-pixel
       round-up doesn't expose a gap. */
    position: sticky;
    top: 38px;
    z-index: 2;
    border-bottom: 1px solid #e4e4e8;
}

.sidebar-header {
    padding: 8px 10px;
    background: #f4f4f6;
    font-size: 13px;
    color: #333;
    display: flex;
    align-items: center;
    gap: 6px;
}
.sidebar-header strong {
    font-weight: 600;
}
.sidebar-header span#conformer-count {
    color: #888;
    font-size: 12px;
}
.sidebar-header .spacer {
    flex: 1;
}

/* Segmented toggle pill (energy unit, chart axis). */
.seg-toggle {
    display: inline-flex;
    border: 1px solid #c8c8d0;
    border-radius: 4px;
    overflow: hidden;
    font-size: 11px;
}
.seg-toggle button {
    background: #fff;
    border: none;
    padding: 3px 7px;
    cursor: pointer;
    color: #555;
    font-size: 11px;
    line-height: 1.2;
    border-right: 1px solid #c8c8d0;
}
.seg-toggle button:last-child {
    border-right: none;
}
.seg-toggle button.active {
    background: #2152a4;
    color: #fff;
}
.seg-toggle.small button {
    padding: 2px 6px;
    font-size: 10px;
}

/* Sparkline block. The SVG's viewBox aspect (2.8:1) drives its
   rendered height — no fixed CSS height — so circles stay
   circular when the sidebar resizes. */
.sparkline-row {
    padding: 6px 8px 0;
    background: #fafafb;
}
#sparkline {
    width: 100%;
    height: auto;
    display: block;
}
.sparkline-toolbar {
    padding: 4px 10px 8px;
    background: #fafafb;
    display: flex;
    align-items: center;
    gap: 8px;
}
#sparkline-range {
    color: #888;
    font-size: 11px;
    font-variant-numeric: tabular-nums;
    margin-left: auto;
}

#conformer-list {
    flex: 1;
}

.conformer-card {
    display: flex;
    align-items: stretch;
    cursor: pointer;
    border-bottom: 1px solid #e8e8ec;
    background: #fff;
    transition: background 0.12s;
}
.conformer-card:hover {
    background: #f0f4fa;
}
.conformer-card.active {
    background: #e2ecfb;
}
.conformer-card.active .card-label {
    font-weight: 600;
}

.color-band {
    width: 4px;
    flex: 0 0 4px;
}

.card-body {
    flex: 1;
    padding: 8px 10px;
    min-width: 0;
}

.card-label {
    font-size: 13px;
    color: #222;
    margin-bottom: 3px;
}

/* Energy badge — small framed monospace box, same visual treatment
   as the SMILES box up top. The boxed bg/border gives the
   .flash-copied green-flash overlay a visible target. */
.card-dE {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 11px;
    color: #333;
    background: #f6f6f9;
    border: 1px solid #e8e8ec;
    border-radius: 3px;
    padding: 2px 6px;
    margin-bottom: 5px;
    display: inline-block;
    font-variant-numeric: tabular-nums;
    line-height: 1.35;
}

.bar-outer {
    height: 4px;
    background: #ececf0;
    border-radius: 2px;
    overflow: hidden;
}
.bar-inner {
    height: 100%;
    border-radius: 2px;
}

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
    z-index: 100;
}
/* The HTML5 ``hidden`` global attribute relies on the UA's
   ``display: none`` rule. ID-targeted ``display`` properties below
   have higher specificity (ID > attribute) and would otherwise win,
   leaving the overlay permanently visible on top of the viewer.
   These explicit overrides re-establish hidden-respect. */
#dropzone[hidden],
#structure-2d[hidden] {
    display: none;
}

/* 2D structure inset. Three visual states:
   (a) corner   — 80×80, bottom-left of viewer (default)
   (b) hover    — 240×240 inline expansion (only when not .expanded)
   (c) expanded — large overlay covering the viewer, for picking
                  substructures comfortably
   The cubic-bezier on the size + position properties uses the same
   timing across all three so the inset feels physically anchored
   as it grows / shrinks. */
#structure-2d {
    position: absolute;
    bottom: 12px;
    left: 12px;
    width: 80px;
    height: 80px;
    background: rgba(255, 255, 255, 0.95);
    border: 1px solid #d8d8dc;
    border-radius: 6px;
    overflow: hidden;
    cursor: pointer;
    transition: width 0.28s cubic-bezier(.4,.0,.2,1),
                height 0.28s cubic-bezier(.4,.0,.2,1),
                top 0.28s cubic-bezier(.4,.0,.2,1),
                left 0.28s cubic-bezier(.4,.0,.2,1),
                bottom 0.28s cubic-bezier(.4,.0,.2,1),
                right 0.28s cubic-bezier(.4,.0,.2,1),
                box-shadow 0.22s ease,
                border-color 0.22s ease;
    z-index: 50;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06);
}
#structure-2d:not(.expanded):hover {
    width: 240px;
    height: 240px;
    border-color: #b8b8c0;
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.18);
}
#structure-2d.expanded {
    /* Use viewport-relative insets so the inset stays centered over
       the 3D viewer no matter what size it is. The 3D viewer's
       interaction is paused while expanded (overlay catches input). */
    top: 16px;
    right: 16px;
    bottom: 16px;
    left: 16px;
    width: auto;
    height: auto;
    border-color: #b8b8c0;
    box-shadow: 0 12px 36px rgba(0, 0, 0, 0.22);
}

/* SVG host inside #structure-2d. RDKit's get_svg_with_highlights()
   returns a self-contained <svg> we innerHTML in here; CSS makes
   the SVG fill the host so it scales with the size transitions. */
#structure-canvas {
    width: 100%;
    height: 100%;
    display: block;
    overflow: hidden;
}
#structure-canvas svg {
    width: 100%;
    height: 100%;
    display: block;
    background: transparent;
}
/* RDKit emits ``class="atom-N"`` on heteroatom labels and
   ``class="bond-N atom-A atom-B"`` on bond paths. Surface a copy
   cursor over both so the user knows they're clickable; the
   delegation handler on #structure-canvas does the rest. */
#structure-canvas [class*="atom-"],
#structure-canvas [class*="bond-"] {
    cursor: pointer;
}

/* Chevron toggle in the inset's top-right. Only opaque on hover
   (corner state) or always-on when .expanded. */
#structure-expand-btn {
    position: absolute;
    top: 4px;
    right: 4px;
    width: 22px;
    height: 22px;
    border: 1px solid #d8d8dc;
    border-radius: 4px;
    background: rgba(255, 255, 255, 0.92);
    color: #555;
    font-size: 13px;
    line-height: 1;
    padding: 0;
    cursor: pointer;
    opacity: 0;
    transition: opacity 0.18s ease,
                background 0.15s ease,
                color 0.15s ease;
    z-index: 2;
}
#structure-2d:hover #structure-expand-btn,
#structure-2d.expanded #structure-expand-btn {
    opacity: 1;
}
#structure-expand-btn:hover {
    background: #2152a4;
    color: #fff;
    border-color: #2152a4;
}

/* Bottom toolbar — Align / Reset / "N selected" indicator. Hidden
   in corner state, slides in on hover or expanded. */
#structure-toolbar {
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    padding: 6px 8px;
    background: linear-gradient(to top,
                                rgba(245,245,250,0.96) 60%,
                                rgba(245,245,250,0));
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 12px;
    opacity: 0;
    transform: translateY(4px);
    transition: opacity 0.18s ease, transform 0.18s ease;
    pointer-events: none;
    z-index: 2;
}
#structure-2d:hover #structure-toolbar,
#structure-2d.expanded #structure-toolbar {
    opacity: 1;
    transform: translateY(0);
    pointer-events: auto;
}
#structure-selection-count {
    flex: 1;
    color: #555;
    font-variant-numeric: tabular-nums;
}
.sb-action {
    background: #fff;
    border: 1px solid #c8c8d0;
    color: #444;
    border-radius: 4px;
    padding: 3px 10px;
    cursor: pointer;
    font-size: 12px;
    line-height: 1.3;
    transition: background 0.12s ease,
                border-color 0.12s ease,
                color 0.12s ease;
}
.sb-action:hover {
    background: #f4f6fa;
    border-color: #9aa6c4;
}
.sb-action.align {
    background: #2152a4;
    color: #fff;
    border-color: #2152a4;
}
.sb-action.align:hover {
    background: #1a4488;
}
.sb-action.align[disabled],
.sb-action.reset[disabled] {
    opacity: 0.45;
    cursor: not-allowed;
}

/* Copy-to-clipboard visual feedback. ``.flash-copied`` is added by
   copy_value() for ~700ms after a successful clipboard write.
   The "Copied!" pill sits BELOW the element and is horizontally
   centered — works for the wide SMILES box (which sits near the
   right edge of the sidebar) and the narrow energy boxes (which
   sit on the left side of conformer cards) without overflowing
   the sidebar on either side. */
.copy-target {
    cursor: copy;
    position: relative;
}
.flash-copied {
    background: #c8e6c9 !important;
    border-color: #4caf50 !important;
    box-shadow: 0 0 0 1px #4caf50, 0 0 8px rgba(76, 175, 80, 0.35);
    transition: background 0.1s ease, border-color 0.1s ease,
                box-shadow 0.1s ease;
}
.flash-copied::after {
    content: 'Copied!';
    position: absolute;
    top: calc(100% + 4px);
    left: 50%;
    transform: translateX(-50%);
    background: #4caf50;
    color: #fff;
    padding: 1px 6px;
    border-radius: 3px;
    font-size: 10px;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
    font-weight: 500;
    pointer-events: none;
    white-space: nowrap;
    z-index: 1000;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15);
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
# Manifest builder (identical shape to geometry_viewer)
# ---------------------------------------------------------------------------


def _make_block_key() -> str:
    hex_part = secrets.token_hex(7)
    num_part = secrets.randbelow(10**8)
    return f"onlb_{hex_part}.{num_part:08d}"


def _make_file_entry(
    file_info_id: int, node_id: int, block_key: str,
    file_name: str, file_format: str, rel_path: str, entry_point: bool,
) -> dict:
    return {
        "node_files_info_id": file_info_id,
        "node_id": node_id,
        "file_name": file_name,
        "file_format": file_format,
        "upload_method": "code",
        "path": rel_path,
    }


def _make_layout_file_entry(
    file_info_id: int, node_id: int, block_key: str,
    file_name: str, file_format: str, rel_path: str, entry_point: bool,
) -> dict:
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
    base_dir = f"{node_id}/{block_key}"
    files_meta = [
        (1, "index.html", "unknown",    f"{base_dir}/index.html",     True),
        (2, "viewer.js",  "javascript", f"{base_dir}/js/viewer.js",   False),
        (3, "styles.css", "txt",        f"{base_dir}/css/styles.css", False),
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
        # Single inport — the multi-frame xyz file.
        "inputs": [
            {
                "node_input_id": 1,
                "node_id": node_id,
                "name": "ensemble_file",
                "type": "text",
                "tags": None,
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
    return hashlib.sha256(payload).hexdigest()[:13]


def write_zip(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    node_id = PLACEHOLDER_NODE_ID
    block_key = _make_block_key()
    manifest = build_manifest(node_id, block_key)
    manifest_bytes = json.dumps(manifest, indent=2).encode("utf-8")

    zip_path = out_dir / f"NODE_{NODE_NAME}_{_short_hash(manifest_bytes)}.zip"

    base_dir = f"{node_id}/{block_key}"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"{base_dir}/index.html",   INDEX_HTML)
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
    print("Upload via the GUI's Node Manager. In the workflow editor,")
    print(f"wire wf-prism's ``xyz_ensemble`` output (its accepted_ensemble.xyz)")
    print(f"→ {NODE_NAME}.ensemble_file. Run the protocol; the viewer iframe")
    print("will appear once prism_screen finishes.")


if __name__ == "__main__":
    main()
