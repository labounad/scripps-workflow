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
