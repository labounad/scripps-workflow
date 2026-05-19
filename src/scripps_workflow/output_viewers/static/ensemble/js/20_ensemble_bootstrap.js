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
        atom_map: Array.isArray(window.__pending_atom_map) ? window.__pending_atom_map : null,
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
        // 3D click-selection / measurement overlay state. Holds XYZ
        // atom indices for the active molecule; coordinates are looked
        // up fresh from the active 3Dmol model on every render so
        // conformer switches and alignments stay synchronized.
        measurement_atoms: [],
        measurement_shapes: [],
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
