// ---- Measurement state + atom utilities ---------------------------

var VDW_RADII = {
    H: 1.20, HE: 1.40,
    C: 1.70, N: 1.55, O: 1.52, F: 1.47,
    P: 1.80, S: 1.80, CL: 1.75,
    BR: 1.85, I: 1.98
};

function active_model_atoms() {
    var state = window.__viewer_state;
    if (!state || !state.viewer) return [];
    var model = state.viewer.getModel(state.active_idx);
    return model ? model.selectedAtoms({}) : [];
}

function atom_elem_key(atom) {
    if (!atom || !atom.elem) return 'C';
    return String(atom.elem).toUpperCase();
}

function vdw_radius_for_atom(atom) {
    var key = atom_elem_key(atom);
    return VDW_RADII[key] || 1.70;
}

function current_measurement_style() {
    var state = window.__viewer_state;
    return (state && state.style) ? state.style : 'ballstick';
}

function display_atom_radius(atom) {
    var r = vdw_radius_for_atom(atom);
    switch (current_measurement_style()) {
        case 'sphere':
            return r;
        case 'stick':
            return 0.18;
        case 'line':
            return 0.12;
        case 'ballstick':
        default:
            return r * 0.25;
    }
}

function measurement_sphere_radius(atom) {
    var base = display_atom_radius(atom);
    var style = current_measurement_style();
    // Use an additive margin instead of a multiplicative scale, so
    // heavy atoms do not balloon too much relative to hydrogens.
    if (style === 'sphere') {
        return base + 0.12;
    }
    if (style === 'line') {
        return Math.max(0.26, base + 0.14);
    }
    if (style === 'stick') {
        return Math.max(0.32, base + 0.16);
    }
    // Ball-and-stick default. Rendered balls are vdw*0.25; add a
    // small constant halo so C/O/N remain visible without becoming huge.
    return Math.max(0.34, base + 0.10);
}

function measurement_bond_radius() {
    switch (current_measurement_style()) {
        case 'line':
            return 0.07;
        case 'sphere':
            return 0.16;
        case 'stick':
            return 0.22;
        case 'ballstick':
        default:
            return 0.24;
    }
}

function atom_xyz_index_from_click(atom) {
    // Prefer locating by coordinates in the active model so we get a
    // stable zero-based XYZ index even when 3Dmol's serial convention
    // differs between formats. Fall back to index/serial if needed.
    var atoms = active_model_atoms();
    var best = -1;
    var best_d2 = Infinity;
    for (var i = 0; i < atoms.length; i++) {
        var a = atoms[i];
        if (atom.elem && a.elem && atom.elem !== a.elem) continue;
        var dx = (a.x || 0) - (atom.x || 0);
        var dy = (a.y || 0) - (atom.y || 0);
        var dz = (a.z || 0) - (atom.z || 0);
        var d2 = dx*dx + dy*dy + dz*dz;
        if (d2 < best_d2) {
            best = i;
            best_d2 = d2;
        }
    }
    if (best >= 0 && best_d2 < 1e-8) return best;
    if (Number.isFinite(atom.index)) return atom.index;
    if (Number.isFinite(atom.serial)) {
        return atom.serial > 0 ? atom.serial - 1 : atom.serial;
    }
    return -1;
}

function atom_label_for_index(idx) {
    var atoms = active_model_atoms();
    var a = atoms[idx];
    if (!a) return 'Atom' + (idx + 1);
    var serial = Number.isFinite(a.serial) ? a.serial : idx + 1;
    return (a.elem || '?') + serial;
}

function coords_for_measurement_atoms() {
    var state = window.__viewer_state;
    if (!state || !state.measurement_atoms) return [];
    var atoms = active_model_atoms();
    return state.measurement_atoms.map(function(sel) {
        var a = atoms[sel.idx];
        if (!a) return null;
        return {
            idx: sel.idx,
            label: atom_label_for_index(sel.idx),
            coord: vec3(a.x, a.y, a.z),
            atom: a,
        };
    }).filter(function(x) { return !!x; });
}
