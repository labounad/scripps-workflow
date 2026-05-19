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
