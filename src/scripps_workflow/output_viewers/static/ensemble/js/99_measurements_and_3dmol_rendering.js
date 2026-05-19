// ---- 3D atom selection + measurement overlay -----------------------

var MEASUREMENT_COLOR = '#2563eb';
var MEASUREMENT_COLOR_LIGHT = '#93c5fd';
var MEASUREMENT_FILL_COLOR = '#60a5fa';
var MEASUREMENT_LINE_RADIUS = 0.035;
var VDW_RADII = {
    H: 1.20, HE: 1.40,
    C: 1.70, N: 1.55, O: 1.52, F: 1.47,
    P: 1.80, S: 1.80, CL: 1.75,
    BR: 1.85, I: 1.98
};

function vec3(x, y, z) { return { x: x, y: y, z: z }; }
function vsub(a, b) { return vec3(a.x - b.x, a.y - b.y, a.z - b.z); }
function vadd(a, b) { return vec3(a.x + b.x, a.y + b.y, a.z + b.z); }
function vscale(a, s) { return vec3(a.x * s, a.y * s, a.z * s); }
function vdot(a, b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
function vcross(a, b) {
    return vec3(
        a.y*b.z - a.z*b.y,
        a.z*b.x - a.x*b.z,
        a.x*b.y - a.y*b.x
    );
}
function vnorm(a) { return Math.sqrt(vdot(a, a)); }
function vnormalize(a) {
    var n = vnorm(a);
    return n > 1e-12 ? vscale(a, 1 / n) : vec3(0, 0, 0);
}
function vdistance(a, b) { return vnorm(vsub(a, b)); }
function clamp(x, lo, hi) { return Math.max(lo, Math.min(hi, x)); }
function radians_to_degrees(x) { return x * 180 / Math.PI; }
function format_degrees(x) { return (Math.abs(x) < 0.0005 ? 0 : x).toFixed(2) + '\u00b0'; }
function format_distance(x) { return x.toFixed(3) + ' \u00c5'; }

function angle_abc(a, b, c) {
    var ba = vnormalize(vsub(a, b));
    var bc = vnormalize(vsub(c, b));
    return radians_to_degrees(Math.acos(clamp(vdot(ba, bc), -1, 1)));
}

function dihedral_abcd(a, b, c, d) {
    // Conventional signed torsion angle A-B-C-D, in degrees.
    var b0 = vscale(vsub(b, a), -1);
    var b1 = vsub(c, b);
    var b2 = vsub(d, c);
    var b1n = vnormalize(b1);
    var v = vsub(b0, vscale(b1n, vdot(b0, b1n)));
    var w = vsub(b2, vscale(b1n, vdot(b2, b1n)));
    var x = vdot(v, w);
    var y = vdot(vcross(b1n, v), w);
    return radians_to_degrees(Math.atan2(y, x));
}

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

function clear_measurement_shapes() {
    var state = window.__viewer_state;
    if (!state || !state.viewer) return;
    var shapes = state.measurement_shapes || [];
    shapes.forEach(function(shape) {
        try { state.viewer.removeShape(shape); } catch (e) {}
    });
    state.measurement_shapes = [];
}

function add_measurement_shape(shape) {
    var state = window.__viewer_state;
    if (!state) return shape;
    if (!state.measurement_shapes) state.measurement_shapes = [];
    if (shape) state.measurement_shapes.push(shape);
    return shape;
}

function add_highlight_sphere(viewer, item) {
    return add_measurement_shape(viewer.addSphere({
        center: item.coord,
        radius: measurement_sphere_radius(item.atom),
        color: MEASUREMENT_COLOR,
        opacity: 0.42,
        alpha: 0.42,
    }));
}

function add_cylinder_segment(viewer, start, end, radius, color, opacity) {
    return add_measurement_shape(viewer.addCylinder({
        start: start,
        end: end,
        radius: radius || MEASUREMENT_LINE_RADIUS,
        color: color || MEASUREMENT_COLOR,
        opacity: opacity == null ? 0.50 : opacity,
        alpha: opacity == null ? 0.50 : opacity,
        fromCap: 1,
        toCap: 1,
    }));
}

function atoms_are_bonded(idx_a, idx_b) {
    var atoms = active_model_atoms();
    var a = atoms[idx_a];
    if (!a || !Array.isArray(a.bonds)) return false;
    return a.bonds.indexOf(idx_b) >= 0;
}

function add_bond_highlight(viewer, a, b) {
    return add_measurement_shape(viewer.addCylinder({
        start: a,
        end: b,
        radius: measurement_bond_radius(),
        color: MEASUREMENT_COLOR,
        opacity: 0.34,
        alpha: 0.34,
        fromCap: 1,
        toCap: 1,
    }));
}

function add_dashed_line(viewer, a, b, opts) {
    opts = opts || {};
    var gap_fraction = opts.gap_fraction || 0.42;
    var dash_count = opts.dash_count || Math.max(4, Math.ceil(vdistance(a, b) / 0.35));
    var radius = opts.radius || MEASUREMENT_LINE_RADIUS;
    var color = opts.color || MEASUREMENT_COLOR;
    var opacity = opts.opacity == null ? 0.48 : opts.opacity;
    var ab = vsub(b, a);
    for (var i = 0; i < dash_count; i++) {
        var t0 = i / dash_count;
        var t1 = (i + 1 - gap_fraction) / dash_count;
        add_cylinder_segment(
            viewer,
            vadd(a, vscale(ab, t0)),
            vadd(a, vscale(ab, t1)),
            radius,
            color,
            opacity
        );
    }
}

function add_measurement_connection(viewer, item_a, item_b) {
    if (atoms_are_bonded(item_a.idx, item_b.idx)) {
        add_bond_highlight(viewer, item_a.coord, item_b.coord);
    } else {
        add_dashed_line(viewer, item_a.coord, item_b.coord);
    }
}

function add_angle_wedge(viewer, a, b, c) {
    var ba = vnormalize(vsub(a, b));
    var bc = vnormalize(vsub(c, b));
    if (vnorm(ba) < 1e-9 || vnorm(bc) < 1e-9) return;
    var theta = Math.acos(clamp(vdot(ba, bc), -1, 1));
    if (!Number.isFinite(theta) || theta < 1e-6) return;

    var r = Math.min(1.70, Math.max(0.82, 0.64 * Math.min(vdistance(a, b), vdistance(c, b))));
    var normal = vnormalize(vcross(ba, bc));
    if (vnorm(normal) < 1e-9) return;
    var steps = Math.max(24, Math.ceil(theta / (Math.PI / 48)));

    function dir_at(t) {
        return vnormalize(vadd(
            vscale(ba, Math.cos(t)),
            vscale(vcross(normal, ba), Math.sin(t))
        ));
    }

    var verts = [b];
    for (var i = 0; i <= steps; i++) {
        verts.push(vadd(b, vscale(dir_at(theta * i / steps), r)));
    }

    // Preferred true filled sector mesh.
    // 3Dmol expects a flat integer face array; include both windings so the
    // transparent wedge is visible from either side of the molecular plane.
    var mesh_ok = false;
    try {
        var vertexArr = verts.map(function(v) { return new $3Dmol.Vector3(v.x, v.y, v.z); });
        var faceArr = [];
        for (var j = 1; j < verts.length - 1; j++) {
            faceArr.push(0, j, j + 1);
            faceArr.push(0, j + 1, j);
        }
        add_measurement_shape(viewer.addCustom({
            vertexArr: vertexArr,
            faceArr: faceArr,
            color: MEASUREMENT_FILL_COLOR,
            opacity: 0.28,
            alpha: 0.28,
        }));
        mesh_ok = true;
    } catch (err) {
        mesh_ok = false;
    }

    // Fallback fill: dense translucent radial strokes from the vertex to the
    // outer arc. This reads like a shaded sector without creating an extra
    // dashed/segmented arc border.
    if (!mesh_ok) {
        for (var f = 1; f < verts.length; f++) {
            add_cylinder_segment(viewer, b, verts[f], 0.010, MEASUREMENT_FILL_COLOR, 0.075);
        }
    }

    // Boundary spokes and a single solid outer arc.
    add_cylinder_segment(viewer, b, verts[1], 0.026, MEASUREMENT_COLOR_LIGHT, 0.55);
    add_cylinder_segment(viewer, b, verts[verts.length - 1], 0.026, MEASUREMENT_COLOR_LIGHT, 0.55);
    for (var k = 1; k < verts.length - 1; k++) {
        add_cylinder_segment(viewer, verts[k], verts[k + 1], 0.030, MEASUREMENT_COLOR_LIGHT, 0.68);
    }
}

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
    toggle_measurement_atom(atom);
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
    render_measurement_overlay();
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
