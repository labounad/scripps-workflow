// ---- 3D measurement shape rendering -------------------------------

var MEASUREMENT_COLOR = '#2563eb';
var MEASUREMENT_COLOR_LIGHT = '#93c5fd';
var MEASUREMENT_FILL_COLOR = '#60a5fa';
var MEASUREMENT_LINE_RADIUS = 0.035;

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
