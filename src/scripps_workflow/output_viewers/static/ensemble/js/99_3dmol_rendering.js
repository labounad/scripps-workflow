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
