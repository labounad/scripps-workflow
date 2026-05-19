// ---- Measurement UI and selection actions -------------------------

function measurement_html(items) {
    if (!items || items.length === 0) return '';
    var chips = items.map(function(x) {
        return '<span class="measurement-chip">' + x.label + '</span>';
    }).join('<span class="measurement-sep"> </span>');
    var extra = '';
    if (items.length === 2) {
        extra = '<span class="measurement-value">d = ' + format_distance(vdistance(items[0].coord, items[1].coord)) + '</span>';
    } else if (items.length === 3) {
        extra = '<span class="measurement-value">\u2220 ' + items[0].label + '-' + items[1].label + '-' + items[2].label
            + ' = ' + format_degrees(angle_abc(items[0].coord, items[1].coord, items[2].coord)) + '</span>';
    } else if (items.length === 4) {
        extra = '<span class="measurement-value">dihedral ' + items[0].label + '-' + items[1].label + '-' + items[2].label + '-' + items[3].label
            + ' = ' + format_degrees(dihedral_abcd(items[0].coord, items[1].coord, items[2].coord, items[3].coord)) + '</span>';
    }
    return chips + (extra ? '<span class="measurement-sep">|</span>' + extra : '');
}

function render_measurement_overlay() {
    var state = window.__viewer_state;
    if (!state || !state.viewer) return;
    clear_measurement_shapes();

    var pin_el = document.getElementById('atom-pinned-label');
    var items = coords_for_measurement_atoms();
    if (!pin_el) return;

    if (items.length === 0) {
        pin_el.classList.remove('visible');
        pin_el.innerHTML = '';
        return;
    }

    if (items.length >= 2) {
        add_measurement_connection(state.viewer, items[0], items[1]);
    }
    if (items.length >= 3) {
        add_measurement_connection(state.viewer, items[1], items[2]);
        add_angle_wedge(state.viewer, items[0].coord, items[1].coord, items[2].coord);
    }
    if (items.length >= 4) {
        add_measurement_connection(state.viewer, items[2], items[3]);
        add_angle_wedge(state.viewer, items[1].coord, items[2].coord, items[3].coord);
    }
    // Draw spheres last so they remain visibly on top of the
    // connection overlays.
    items.forEach(function(x) { add_highlight_sphere(state.viewer, x); });

    pin_el.innerHTML = measurement_html(items);
    pin_el.classList.add('visible');
}

function toggle_measurement_atom(atom) {
    var state = window.__viewer_state;
    if (!state || !atom) return;
    var idx = atom_xyz_index_from_click(atom);
    if (idx < 0) return;
    if (!Array.isArray(state.measurement_atoms)) state.measurement_atoms = [];

    var existing = state.measurement_atoms.findIndex(function(x) { return x.idx === idx; });
    if (existing >= 0) {
        state.measurement_atoms.splice(existing, 1);
    } else {
        if (state.measurement_atoms.length >= 4) {
            state.measurement_atoms = [];
        }
        state.measurement_atoms.push({ idx: idx });
    }
    render_measurement_overlay();
    state.viewer.render();
}
