// ensemble_viewer — 3Dmol.js multi-conformer viewer.
//
// Standalone bundle mode: workflow data are embedded in index.html.
// Optional developer mode: ?xyz=<url> works when served over HTTP.
//
// The multi-frame XYZ is parsed into per-conformer frames.
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
        smiles: 'CC/C=C\\CC1=C(CCC1=O)C',  // L2 test fixture
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
    window.__pending_smiles = params.get('smiles') || null;
    window.__pending_atom_map = null;
    window.__viewer_data_loaded = false;

    install_keyboard_handler();
    kick_off_rdkit();

    // Standalone bundle path: all workflow data are embedded directly in
    // index.html by the Python bundle builder. This is intentionally the
    // default path because local file:// pages are not allowed to fetch
    // sibling files or re-fetch their own index.html in modern browsers.
    var payload_el = document.getElementById('scripps-viewer-input') ||
                     document.getElementById('viewer-input');
    if (payload_el) {
        try {
            var meta = JSON.parse(payload_el.textContent || '{}');
            if (meta.smiles && !window.__pending_smiles) {
                window.__pending_smiles = meta.smiles;
            }
            if (load_embedded_payload(meta, 'embedded standalone')) {
                return;
            }
        } catch (err) {
            console.error('Embedded viewer payload could not be parsed:', err);
            set_status('Could not parse embedded viewer payload: ' + err, true);
            return;
        }
    }

    // Developer/local-server harness path. This intentionally requires an
    // HTTP server; browsers block fetch(file://...) from file:// pages.
    if (params.get('xyz')) {
        return resolve_local_test(params);
    }

    set_status(
        'No embedded ensemble payload found in index.html. Rebuild the viewer bundle, '
        + 'or serve this directory with a local HTTP server and pass ?xyz=...',
        true
    );
}
function load_embedded_payload(meta, source_label) {
    if (!meta) return false;
    if (meta.smiles && !window.__pending_smiles) {
        window.__pending_smiles = meta.smiles;
    }
    var atom_map = meta.atom_map || (meta.metadata && meta.metadata.atom_map);
    if (Array.isArray(atom_map)) {
        window.__pending_atom_map = atom_map;
    }
    var xyz_text = meta.xyz_text || meta.xyz || meta.text;
    if (!xyz_text) return false;
    try {
        set_status('loading ' + source_label + ' ensemble data');
        init_ensemble(String(xyz_text));
        window.__viewer_data_loaded = true;
        return true;
    } catch (err) {
        console.error('Failed to initialize embedded ensemble:', err);
        set_status('Failed to initialize embedded ensemble: ' + err, true);
        return false;
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
function resolve_local_test(params) {
    var url = params.get('xyz');
    if (window.location.protocol === 'file:') {
        set_status('The ?xyz= developer mode requires serving this directory over HTTP; file:// pages cannot fetch local files.', true);
        return;
    }
    set_status('local-test mode: fetching ' + url);
    fetch_xyz(url);
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
    console.log('[ensemble_viewer] fetching xyz:', url);
    return fetch(url)
        .then(function(response) {
            if (!response.ok) throw new Error('HTTP ' + response.status + ' while fetching ' + url);
            return response.text();
        })
        .then(function(text) {
            console.log('[ensemble_viewer] fetched xyz bytes:', text.length);
            init_ensemble(text);
            window.__viewer_data_loaded = true;
            return true;
        })
        .catch(function(err) {
            console.error(err);
            set_status('Error fetching xyz: ' + err, true);
            throw err;
        });
}
