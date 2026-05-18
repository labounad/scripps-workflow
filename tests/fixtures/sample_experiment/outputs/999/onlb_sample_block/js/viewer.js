// ensemble_viewer — 3Dmol.js multi-conformer viewer.
//
// Resolution modes (same shape as geometry_viewer):
//   A. LOCAL_TEST: ?xyz=<absolute-or-relative-url> [&smiles=<str>]
//   B. WORKFLOW_GUI: ?experiment_id=&protocol_id=&protocol_name=&file_name=
//                   [&smiles=<str>]
//   C. GUI_INPORT: ?protocol_id=&user_id=
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

var OPAAT_LEGACY_SERVICE = 'http://opaat.scripps.edu/workflow_webapp_api/services.php?serviceName=get_inputs_for_output_node';
var OPAAT_RESOURCE_PREFIX = 'http://opaat.scripps.edu/';

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
    var mode = pick_resolution_mode(params);
    console.log('ensemble_viewer resolution mode:', mode);

    // The SMILES is orthogonal to the resolution mode — any mode can
    // optionally carry it via ?smiles=<encoded>. Stash early so it's
    // available when init_ensemble fires later. A staged viewer manifest
    // may override this below when an upstream bridge node copied data
    // directly into this output block.
    window.__pending_smiles = params.get('smiles') || null;
    window.__viewer_data_loaded = false;

    install_keyboard_handler();
    // Kick off RDKit.js WASM init in parallel with the xyz fetch.
    // It typically takes ~500ms; render_2d_structure() waits for both
    // RDKit and the viewer state to be ready before drawing the inset.
    kick_off_rdkit();

    // Local/manual modes should behave like ordinary static pages.
    if (mode === 'LOCAL_TEST') return resolve_local_test(params);
    if (mode === 'WORKFLOW_GUI') return resolve_workflow_gui(params);

    // Robust deployed-GUI path. First use data embedded/staged by
    // wf-extract-conformers. This avoids browser-side auth/CORS issues
    // entirely when the output HTML/data are available. If no staged
    // payload is present, fall back to the platform's legacy resolver
    // only when explicitly enabled.
    return wait_for_viewer_data(params, { interval_ms: 2000, timeout_ms: 30000 })
        .then(function(loaded) {
            if (loaded) return;
            // The static HTML served by the GUI may be the original imported
            // output-node asset, not the run-directory copy patched by
            // wf-extract-conformers. In that case staged/embedded data will
            // never appear in this iframe. Fall back to the platform inport
            // resolver and keep polling; wf-extract-conformers also schedules
            // delayed re-registration to overwrite the GUI's stale -download
            // output step.
            return wait_for_platform_inport(params, {
                interval_ms: 5000,
                timeout_ms: 600000,
                kind: 'ensemble'
            });
        });
}

function cache_bust(url) {
    var sep = url.indexOf('?') === -1 ? '?' : '&';
    return url + sep + '_=' + Date.now();
}

function load_embedded_payload(meta, source_label) {
    if (!meta) return false;
    if (meta.smiles && !window.__pending_smiles) {
        window.__pending_smiles = meta.smiles;
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

function resolve_embedded_viewer_data() {
    var el = document.getElementById('scripps-viewer-input');
    if (!el) return Promise.resolve(false);
    try {
        return Promise.resolve(load_embedded_payload(
            JSON.parse(el.textContent || '{}'),
            'embedded'
        ));
    } catch (err) {
        console.warn('Embedded viewer payload could not be parsed:', err);
        return Promise.resolve(false);
    }
}

function resolve_embedded_viewer_data_from_html(html_text) {
    try {
        var doc = new DOMParser().parseFromString(String(html_text), 'text/html');
        var el = doc.getElementById('scripps-viewer-input');
        if (!el) return false;
        var meta = JSON.parse(el.textContent || '{}');
        return load_embedded_payload(meta, 'refreshed embedded');
    } catch (err) {
        console.warn('Could not parse refreshed index.html for viewer payload:', err);
        return false;
    }
}

function resolve_refreshed_index_payload() {
    var url = window.location.href.split('#')[0];
    return fetch(cache_bust(url), { cache: 'no-store', credentials: 'same-origin' })
        .then(function(r) {
            if (!r.ok) return false;
            return r.text();
        })
        .then(function(text) {
            if (!text) return false;
            return resolve_embedded_viewer_data_from_html(text);
        })
        .catch(function(err) {
            console.warn('Could not refresh viewer index while waiting:', err);
            return false;
        });
}

function resolve_staged_viewer_data() {
    return fetch(cache_bust('data/viewer_input.json'), {
            cache: 'no-store',
            credentials: 'same-origin',
        })
        .then(function(r) {
            if (!r.ok) return false;
            return r.json();
        })
        .then(function(meta) {
            if (!meta) return false;
            if (meta.smiles && !window.__pending_smiles) {
                window.__pending_smiles = meta.smiles;
            }
            var file_name = meta.file || meta.file_name || 'ensemble.xyz';
            var url = cache_bust('data/' + encodeURIComponent(file_name));
            return fetch(url, { cache: 'no-store', credentials: 'same-origin' })
                .then(function(r) {
                    if (!r.ok) return false;
                    return r.text();
                })
                .then(function(text) {
                    if (!text) return false;
                    try {
                        set_status('loading staged ensemble data');
                        init_ensemble(text);
                        window.__viewer_data_loaded = true;
                        return true;
                    } catch (err) {
                        console.error('Failed to initialize staged ensemble:', err);
                        set_status('Failed to initialize staged ensemble: ' + err, true);
                        return false;
                    }
                });
        })
        .catch(function(err) {
            console.warn('No staged viewer data found yet:', err);
            return false;
        });
}

function wait_for_viewer_data(params, opts) {
    var started = Date.now();
    var interval_ms = opts.interval_ms || 2000;
    var timeout_ms = opts.timeout_ms || 600000;

    function attempt() {
        if (window.__viewer_data_loaded) return Promise.resolve(true);
        return resolve_embedded_viewer_data()
            .then(function(loaded) {
                if (loaded) return true;
                return resolve_staged_viewer_data();
            })
            .then(function(loaded) {
                if (loaded) return true;
                return resolve_refreshed_index_payload();
            })
            .then(function(loaded) {
                if (loaded) return true;
                if (Date.now() - started >= timeout_ms) return false;
                var elapsed = Math.round((Date.now() - started) / 1000);
                set_status('Waiting for ensemble data from wf-extract-conformers... ' + elapsed + 's');
                return new Promise(function(resolve) {
                    setTimeout(function() { resolve(attempt()); }, interval_ms);
                });
            });
    }

    return attempt();
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

function workflow_inport_service_url() {
    // Prefer the workflow GUI backend on the SAME ORIGIN as the
    // iframe. This avoids the browser CORS/mixed-origin failures that
    // occur when Layout nodes try to resolve inports through the old
    // opaat.scripps.edu endpoint.
    var origin = window.location.origin || 'https://workflow.scripps.edu';
    return origin.replace(/\/$/, '')
        + '/workflow_backend/exp_services.php'
        + '?serviceName=get_inputs_for_output_node';
}

function build_inport_resolver_url(service_url, params) {
    var url = service_url
        + '&protocol_id=' + encodeURIComponent(params.get('protocol_id'))
        + '&user_id='     + encodeURIComponent(params.get('user_id'))
        + '&index=0';

    // The workflow backend has evolved over time. Older opaat-style
    // resolvers only needed protocol_id/user_id/index; newer workflow
    // endpoints may also use experiment_id or podid. Forward them when
    // present without making them required.
    ['experiment_id', 'podid'].forEach(function(k) {
        if (params.get(k)) {
            url += '&' + k + '=' + encodeURIComponent(params.get(k));
        }
    });
    return url;
}

function resource_url_from_resolver_payload(data, resource_prefix) {
    if (!data || !data[0] || !data[0].resource_url) {
        return null;
    }
    return new URL(data[0].resource_url, resource_prefix).href;
}

function try_inport_resolver(resolver, params) {
    var url = build_inport_resolver_url(resolver.url, params);
    // Match the existing fasta_viewer node: use a plain GET without
    // credentials. Adding credentials can trigger CORS failures against
    // the legacy opaat endpoint.
    return fetch(url)
        .then(function(r) {
            if (!r.ok) {
                throw new Error(resolver.name + ' HTTP ' + r.status);
            }
            return r.json();
        })
        .then(function(data) {
            var xyz_url = resource_url_from_resolver_payload(
                data,
                resolver.resource_prefix
            );
            if (!xyz_url) {
                throw new Error(resolver.name + ' returned no resource_url');
            }
            return xyz_url;
        });
}

function resolve_inport_with_fallbacks(params) {
    // Do not use workflow_backend from inside the iframe: in practice it
    // returns 401 because the browser-side output iframe does not carry the
    // bearer token used by step_updater.py. The shipped fasta_viewer uses
    // only this legacy opaat resolver, so mirror that contract exactly.
    var resolver = {
        name: 'opaat-legacy',
        url: OPAAT_LEGACY_SERVICE,
        resource_prefix: OPAAT_RESOURCE_PREFIX,
    };
    return try_inport_resolver(resolver, params);
}

function resolve_opaat_legacy(params) {
    set_status('workflow mode: resolving inport');
    resolve_inport_with_fallbacks(params)
        .then(function(xyz_url) {
            fetch_xyz(xyz_url);
        })
        .catch(function(err) {
            console.error(err);
            set_status('Error resolving inport: ' + err.message, true);
        });
}



function wait_for_platform_inport(params, opts) {
    var started = Date.now();
    var interval_ms = opts.interval_ms || 5000;
    var timeout_ms = opts.timeout_ms || 600000;
    var kind = opts.kind || 'ensemble';
    var attempt_no = 0;
    var diagnostic_prefix = '[' + kind + '_viewer] ';

    function summarize_params() {
        return 'href=' + window.location.href
            + ' protocol_id=' + params.get('protocol_id')
            + ' user_id=' + params.get('user_id')
            + ' experiment_id=' + params.get('experiment_id')
            + ' podid=' + params.get('podid');
    }

    function attempt() {
        attempt_no += 1;
        var elapsed = Math.round((Date.now() - started) / 1000);
        set_status('No staged ' + kind + ' data in iframe; waiting for platform inport registration... '
                   + elapsed + 's (attempt ' + attempt_no + ')');
        console.log(diagnostic_prefix + 'platform resolver attempt', attempt_no, summarize_params());
        return resolve_inport_with_fallbacks(params)
            .then(function(xyz_url) {
                console.log(diagnostic_prefix + 'platform resolver returned:', xyz_url);
                set_status('platform inport resolved; loading ' + kind + ' xyz');
                return fetch_xyz(xyz_url);
            })
            .then(function(ok) {
                if (ok) return true;
                throw new Error('fetch_xyz returned false');
            })
            .catch(function(err) {
                console.warn(diagnostic_prefix + 'platform resolver/fetch attempt failed:', err);
                if (Date.now() - started >= timeout_ms) {
                    set_status(
                        'Could not load ' + kind + ' data after ' + elapsed + 's. '
                        + 'Last error: ' + (err && err.message ? err.message : err)
                        + '. Diagnostics: ' + summarize_params(),
                        true
                    );
                    return false;
                }
                return new Promise(function(resolve) {
                    setTimeout(function() { resolve(attempt()); }, interval_ms);
                });
            });
    }

    return attempt();
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

// ---- Multi-frame XYZ parsing ----------------------------------------

function parse_multi_frame_xyz(text) {
    // Returns [{index, n_atoms, comment, text, energy_hartree}].
    // ``text`` per frame is a fully self-contained xyz block
    // (n_atoms\ncomment\natom_lines\n) suitable for 3Dmol.addModel.
    var frames = [];
    var lines = text.split(/\r?\n/);
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
        var frame_text = frame_lines.join('\n') + '\n';

        // ORCA convention: "Coordinates from orca-job orca_opt E -114.299..."
        // CREST writes the energy alone on the line. Match either; fall
        // back to null if neither.
        var energy_hartree = null;
        var m = comment.match(/\bE\s+(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)/);
        if (m) {
            energy_hartree = parseFloat(m[1]);
        } else {
            // CREST-style: just the energy on its own (no "E " prefix).
            var bare = comment.match(/^\s*(-?\d+\.\d+(?:[eE][+-]?\d+)?)\s*$/);
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
            card.title = '\u0394E = ' + (f.delta_kcal !== null
                ? f.delta_kcal.toFixed(3) + ' kcal/mol'
                : 'unknown');
        } else if (f.delta_kcal !== null) {
            dE.textContent = '\u0394E ' + f.delta_kcal.toFixed(2) + ' kcal/mol';
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
        var lines = mb.split('\n');
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
        var nums = d.match(/-?\d+(?:\.\d+)?/g) || [];
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
        if (/(?:^|\s)(atom-hitbox|bond-)/.test(cls)) return;
        var atom_tokens = cls.match(/atom-\d+/g) || [];
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
                : '\u0394E ' + d.value.toFixed(3) + ' kcal/mol');
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
