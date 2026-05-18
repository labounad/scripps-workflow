"""HTML/CSS/JavaScript assets for standalone molecular viewer bundles."""

from __future__ import annotations

THREEDMOL_CDN = "https://3Dmol.org/build/3Dmol-min.js"

ENSEMBLE_INDEX_HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Conformer Ensemble Viewer</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="stylesheet" href="css/styles.css">
  <script src="{THREEDMOL_CDN}"></script>
  <script src="js/viewer.js"></script>
</head>
<body onload="load_page()">
  <div id="toolbar">
    <label>Style:
      <select id="style-select" onchange="apply_style(this.value)">
        <option value="ballstick" selected>Ball &amp; stick</option>
        <option value="stick">Stick</option>
        <option value="sphere">Sphere</option>
        <option value="line">Line</option>
      </select>
    </label>
    <label>View:
      <select id="mode-select" onchange="set_view_mode(this.value)">
        <option value="overlay" selected>Focus + overlay</option>
        <option value="single">Single conformer</option>
      </select>
    </label>
    <button type="button" onclick="reset_view()">Reset zoom</button>
    <span id="status">Loading...</span>
  </div>
  <div id="content">
    <main id="viewer"></main>
    <aside id="sidebar">
      <section id="title-block">
        <div id="viewer-title">Conformer Ensemble</div>
        <div id="viewer-source"></div>
      </section>
      <section id="smiles-block" hidden>
        <div class="label">SMILES</div>
        <pre id="smiles-value"></pre>
      </section>
      <section class="panel">
        <div class="panel-header">
          <strong>Conformers <span id="conformer-count"></span></strong>
          <div class="toggle">
            <button id="unit-kcal" class="active" onclick="set_energy_unit('kcal')">ΔE kcal/mol</button>
            <button id="unit-hartree" onclick="set_energy_unit('hartree')">E hartree</button>
          </div>
        </div>
        <svg id="sparkline" viewBox="0 0 280 80" preserveAspectRatio="none"></svg>
      </section>
      <div id="conformer-list"></div>
    </aside>
  </div>
  <script id="viewer-input" type="application/json">__VIEWER_INPUT__</script>
</body>
</html>
"""

ENSEMBLE_VIEWER_JS = r"""
var HARTREE_TO_KCAL_PER_MOL = 627.5094740631;
var KT_KCAL_298K = 0.592484;
var state = null;

function $(id) { return document.getElementById(id); }

function set_status(text, is_error) {
  var el = $('status');
  if (!el) return;
  el.textContent = text || '';
  el.style.color = is_error ? '#a40000' : '#333';
}

function load_page() {
  if (typeof $3Dmol === 'undefined') {
    set_status('Could not load 3Dmol.js. Check your internet connection.', true);
    return;
  }
  var payload_el = $('viewer-input');
  if (!payload_el) {
    set_status('Missing embedded viewer payload in index.html.', true);
    return;
  }
  var payload;
  try { payload = JSON.parse(payload_el.textContent || '{}'); }
  catch (err) { set_status('Could not parse embedded viewer payload: ' + err, true); return; }
  if (!payload.xyz_text) {
    set_status('Embedded viewer payload does not contain xyz_text.', true);
    return;
  }
  init_ensemble(payload);
}

function parse_multi_frame_xyz(text) {
  var frames = [];
  var lines = String(text).replace(/\r\n/g, '\n').replace(/\r/g, '\n').split('\n');
  var i = 0;
  while (i < lines.length) {
    while (i < lines.length && lines[i].trim() === '') i++;
    if (i >= lines.length) break;
    var n = parseInt(lines[i].trim(), 10);
    if (!Number.isFinite(n) || n <= 0) break;
    if (i + 1 >= lines.length || i + 2 + n > lines.length) break;
    var comment = lines[i + 1] || '';
    var block = lines.slice(i, i + 2 + n).join('\n') + '\n';
    var e = null;
    var m = comment.match(/\bE\s+(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)/);
    if (m) e = parseFloat(m[1]);
    else {
      var b = comment.match(/^\s*(-?\d+\.\d+(?:[eE][+-]?\d+)?)\s*$/);
      if (b) e = parseFloat(b[1]);
    }
    frames.push({index: frames.length + 1, n_atoms: n, comment: comment, text: block, energy_hartree: e});
    i += 2 + n;
  }
  return frames;
}

function init_ensemble(payload) {
  var frames = parse_multi_frame_xyz(payload.xyz_text);
  if (!frames.length) { set_status('No conformers parsed from embedded XYZ.', true); return; }
  var energies = frames.map(function(f) { return f.energy_hartree; }).filter(function(e) { return e !== null && Number.isFinite(e); });
  var min_e = energies.length ? Math.min.apply(null, energies) : null;
  var max_de = 0;
  frames.forEach(function(f) {
    f.delta_kcal = (min_e !== null && f.energy_hartree !== null) ? (f.energy_hartree - min_e) * HARTREE_TO_KCAL_PER_MOL : null;
    if (f.delta_kcal !== null && f.delta_kcal > max_de) max_de = f.delta_kcal;
  });
  var sorted = frames.map(function(_, i) { return i; }).sort(function(a, b) {
    var da = frames[a].delta_kcal, db = frames[b].delta_kcal;
    if (da === null && db === null) return a - b;
    if (da === null) return 1;
    if (db === null) return -1;
    return da - db;
  });
  var exps = frames.map(function(f) { return f.delta_kcal === null ? 0 : Math.exp(-f.delta_kcal / KT_KCAL_298K); });
  var Z = exps.reduce(function(a, b) { return a + b; }, 0) || 1;
  var max_weight = 0;
  frames.forEach(function(f, i) { f.weight = exps[i] / Z; if (f.weight > max_weight) max_weight = f.weight; });
  frames.forEach(function(f) { f.norm = max_de ? Math.min(1, Math.max(0, f.delta_kcal / max_de)) : 0; f.weight_norm = max_weight ? f.weight / max_weight : 0; });

  var viewer = $3Dmol.createViewer($('viewer'), {backgroundColor: 'white'});
  frames.forEach(function(f) { viewer.addModel(f.text, 'xyz'); });
  state = {payload: payload, frames: frames, sorted: sorted, active: sorted[0], viewer: viewer, style: 'ballstick', mode: 'overlay', unit: 'kcal'};
  render_header();
  render_sidebar();
  render_sparkline();
  repaint_viewer();
  viewer.zoomTo();
  viewer.render();
  set_status('');
}

function render_header() {
  $('conformer-count').textContent = '(' + state.frames.length + ')';
  $('viewer-title').textContent = state.payload.title || 'Conformer Ensemble';
  $('viewer-source').textContent = state.payload.source_label || state.payload.source_path || '';
  if (state.payload.smiles) {
    $('smiles-value').textContent = state.payload.smiles;
    $('smiles-block').hidden = false;
  }
}

function energy_text(f) {
  if (state.unit === 'hartree') return f.energy_hartree === null ? 'E unknown' : 'E ' + f.energy_hartree.toFixed(8) + ' Eh';
  return f.delta_kcal === null ? 'ΔE unknown' : 'ΔE ' + f.delta_kcal.toFixed(2) + ' kcal/mol';
}

function color_for(t) { return 'hsl(' + (120 * (1 - t)).toFixed(0) + ', 70%, 45%)'; }

function render_sidebar() {
  $('unit-kcal').classList.toggle('active', state.unit === 'kcal');
  $('unit-hartree').classList.toggle('active', state.unit === 'hartree');
  var list = $('conformer-list');
  list.innerHTML = '';
  state.sorted.forEach(function(idx) {
    var f = state.frames[idx];
    var card = document.createElement('button');
    card.className = 'conformer-card' + (idx === state.active ? ' active' : '');
    card.onclick = function() { set_active(idx); };
    var c = color_for(f.norm || 0);
    var width = Math.max(2, (f.weight_norm || 0) * 100);
    card.innerHTML = '<div class="band" style="background:' + c + '"></div>' +
      '<div class="card-main"><div class="card-title">Conformer ' + f.index + '</div>' +
      '<code>' + energy_text(f) + '</code>' +
      '<div class="bar"><span style="width:' + width.toFixed(1) + '%;background:' + c + '"></span></div></div>';
    list.appendChild(card);
  });
}

function render_sparkline() {
  var svg = $('sparkline');
  svg.innerHTML = '';
  var ns = 'http://www.w3.org/2000/svg';
  var vals = state.sorted.map(function(i) { return state.frames[i].delta_kcal || 0; });
  var maxv = Math.max.apply(null, vals.concat([1]));
  state.sorted.forEach(function(idx, pos) {
    var f = state.frames[idx];
    var x = state.sorted.length === 1 ? 140 : 12 + pos * (256 / (state.sorted.length - 1));
    var y = 68 - ((f.delta_kcal || 0) / maxv) * 55;
    var dot = document.createElementNS(ns, 'circle');
    dot.setAttribute('cx', x); dot.setAttribute('cy', y); dot.setAttribute('r', idx === state.active ? 5 : 3);
    dot.setAttribute('fill', idx === state.active ? '#2563eb' : '#9ca3af');
    dot.style.cursor = 'pointer';
    dot.addEventListener('click', function() { set_active(idx); });
    svg.appendChild(dot);
  });
}

function style_object(style, active) {
  if (style === 'stick') return {stick: {radius: active ? 0.18 : 0.06, opacity: active ? 1 : 0.15}};
  if (style === 'sphere') return {sphere: {scale: active ? 0.35 : 0.12, opacity: active ? 1 : 0.08}};
  if (style === 'line') return {line: {opacity: active ? 1 : 0.18}};
  return active ? {stick: {radius: 0.16}, sphere: {scale: 0.28}} : {line: {opacity: 0.12}};
}

function repaint_viewer() {
  var v = state.viewer;
  v.setStyle({}, {});
  state.frames.forEach(function(_, i) {
    if (state.mode === 'single' && i !== state.active) return;
    v.setStyle({model: i}, style_object(state.style, i === state.active));
  });
  v.render();
}
function set_active(idx) { state.active = idx; render_sidebar(); render_sparkline(); repaint_viewer(); }
function set_view_mode(mode) { state.mode = mode; repaint_viewer(); }
function apply_style(style) { state.style = style; repaint_viewer(); }
function set_energy_unit(unit) { state.unit = unit; render_sidebar(); }
function reset_view() { state.viewer.zoomTo(); state.viewer.render(); }
"""

GEOMETRY_INDEX_HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Geometry Viewer</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="stylesheet" href="css/styles.css">
  <script src="{THREEDMOL_CDN}"></script>
  <script src="js/viewer.js"></script>
</head>
<body onload="load_page()">
  <div id="toolbar">
    <label>Style:
      <select id="style-select" onchange="apply_style(this.value)">
        <option value="ballstick" selected>Ball &amp; stick</option>
        <option value="stick">Stick</option>
        <option value="sphere">Sphere</option>
        <option value="line">Line</option>
      </select>
    </label>
    <button type="button" onclick="reset_view()">Reset zoom</button>
    <span id="status">Loading...</span>
  </div>
  <main id="viewer"></main>
  <aside id="info-panel">
    <h1 id="viewer-title">Geometry Viewer</h1>
    <div class="label">Source</div>
    <pre id="viewer-source"></pre>
    <div id="smiles-block" hidden>
      <div class="label">SMILES</div>
      <pre id="smiles-value"></pre>
    </div>
  </aside>
  <script id="viewer-input" type="application/json">__VIEWER_INPUT__</script>
</body>
</html>
"""

GEOMETRY_VIEWER_JS = r"""
var state = null;
function $(id) { return document.getElementById(id); }
function set_status(text, is_error) {
  var el = $('status');
  if (!el) return;
  el.textContent = text || '';
  el.style.color = is_error ? '#a40000' : '#333';
}
function load_page() {
  if (typeof $3Dmol === 'undefined') {
    set_status('Could not load 3Dmol.js. Check your internet connection.', true);
    return;
  }
  var payload_el = $('viewer-input');
  if (!payload_el) { set_status('Missing embedded viewer payload.', true); return; }
  var payload;
  try { payload = JSON.parse(payload_el.textContent || '{}'); }
  catch (err) { set_status('Could not parse embedded viewer payload: ' + err, true); return; }
  if (!payload.xyz_text) { set_status('Embedded payload does not contain xyz_text.', true); return; }
  var viewer = $3Dmol.createViewer($('viewer'), {backgroundColor: 'white'});
  viewer.addModel(payload.xyz_text, 'xyz');
  state = {payload: payload, viewer: viewer, style: 'ballstick'};
  $('viewer-title').textContent = payload.title || 'Geometry Viewer';
  $('viewer-source').textContent = payload.source_label || payload.source_path || '';
  if (payload.smiles) {
    $('smiles-value').textContent = payload.smiles;
    $('smiles-block').hidden = false;
  }
  repaint_viewer();
  viewer.zoomTo();
  viewer.render();
  set_status('');
}
function style_object(style) {
  if (style === 'stick') return {stick: {radius: 0.18}};
  if (style === 'sphere') return {sphere: {scale: 0.35}};
  if (style === 'line') return {line: {}};
  return {stick: {radius: 0.16}, sphere: {scale: 0.28}};
}
function repaint_viewer() {
  state.viewer.setStyle({}, style_object(state.style));
  state.viewer.render();
}
function apply_style(style) { state.style = style; repaint_viewer(); }
function reset_view() { state.viewer.zoomTo(); state.viewer.render(); }
"""

COMMON_CSS = """
html, body { height: 100%; margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #111827; }
#toolbar { height: 44px; display: flex; align-items: center; gap: 14px; padding: 0 14px; border-bottom: 1px solid #d1d5db; background: #f9fafb; box-sizing: border-box; }
#toolbar select, #toolbar button { font: inherit; padding: 4px 8px; }
#status { margin-left: auto; font-size: 13px; }
#content { display: grid; grid-template-columns: 1fr 360px; height: calc(100% - 44px); }
#viewer { position: relative; min-width: 0; min-height: 0; height: calc(100% - 44px); background: white; }
#content #viewer { height: 100%; }
#sidebar { border-left: 1px solid #d1d5db; background: #fafafa; overflow: auto; }
#sidebar section, .panel { border-bottom: 1px solid #e5e7eb; padding: 12px; }
#title-block { background: white; }
#viewer-title { font-weight: 700; font-size: 18px; margin-bottom: 4px; }
#viewer-source { color: #6b7280; font-size: 12px; overflow-wrap: anywhere; }
.label { font-size: 12px; font-weight: 700; letter-spacing: .04em; color: #6b7280; }
#smiles-value, #viewer-source { white-space: pre-wrap; overflow-wrap: anywhere; background: #f3f4f6; border: 1px solid #e5e7eb; border-radius: 5px; padding: 8px; margin: 6px 0 0; font-size: 12px; }
.panel-header { display: flex; align-items: center; gap: 8px; justify-content: space-between; }
.toggle { display: flex; border: 1px solid #cbd5e1; border-radius: 6px; overflow: hidden; }
.toggle button { border: 0; background: white; padding: 5px 7px; font-size: 12px; cursor: pointer; }
.toggle button.active { background: #1d4ed8; color: white; }
#sparkline { width: 100%; height: 80px; margin-top: 8px; background: #f8fafc; border-radius: 6px; }
#conformer-list { display: flex; flex-direction: column; }
.conformer-card { display: grid; grid-template-columns: 5px 1fr; gap: 0; padding: 0; border: 0; border-bottom: 1px solid #e5e7eb; background: white; text-align: left; cursor: pointer; }
.conformer-card.active { background: #eff6ff; outline: 2px solid #60a5fa; outline-offset: -2px; }
.band { min-height: 70px; }
.card-main { padding: 10px 12px; }
.card-title { font-weight: 600; margin-bottom: 6px; }
.card-main code { background: #f3f4f6; border: 1px solid #e5e7eb; border-radius: 4px; padding: 2px 5px; }
.bar { margin-top: 10px; height: 5px; background: #e5e7eb; border-radius: 99px; overflow: hidden; }
.bar span { display: block; height: 100%; }
#info-panel { position: absolute; top: 58px; right: 14px; width: 320px; max-height: calc(100% - 80px); overflow: auto; background: rgba(255,255,255,.94); border: 1px solid #d1d5db; border-radius: 8px; padding: 14px; box-shadow: 0 12px 30px rgba(15, 23, 42, .12); }
#info-panel h1 { margin: 0 0 12px; font-size: 18px; }
@media (max-width: 900px) { #content { grid-template-columns: 1fr; } #sidebar { display: none; } #info-panel { display: none; } }
"""
