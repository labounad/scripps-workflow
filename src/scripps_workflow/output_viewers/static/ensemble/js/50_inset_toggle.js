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
