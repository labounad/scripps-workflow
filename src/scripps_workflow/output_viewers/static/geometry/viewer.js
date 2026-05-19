
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
