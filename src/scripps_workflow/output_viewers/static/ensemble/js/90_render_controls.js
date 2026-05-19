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
