"""Optional browser-level smoke test for standalone ensemble viewer bundles.

This test is intentionally skipped unless Playwright and a browser binary are
available.  It stubs CDN-loaded 3Dmol/RDKit scripts so the smoke test checks our
bundle bootstrap without requiring public internet access.
"""

from __future__ import annotations

import http.server
import json
import os
import socketserver
import threading
import zipfile
from pathlib import Path

import pytest

from scripps_workflow.output_viewers.ensemble_bundle import build_bundle


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):  # noqa: A002 - matches base class API
        pass


@pytest.mark.browser
def test_ensemble_viewer_bundle_initializes_in_browser(tmp_path, monkeypatch):
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import sync_playwright

    xyz = tmp_path / "ensemble.xyz"
    xyz.write_text(
        "2\nE -1.0\nH 0 0 0\nH 0 0 1\n"
        "2\nE -0.9\nH 0 0 0\nH 0 0 1.1\n",
        encoding="utf-8",
    )
    out = build_bundle(source=str(xyz), smiles="[H][H]", cwd=tmp_path)
    extract_dir = tmp_path / "bundle"
    with zipfile.ZipFile(out) as zf:
        zf.extractall(extract_dir)

    old_cwd = Path.cwd()
    os.chdir(extract_dir)
    server = socketserver.TCPServer(("127.0.0.1", 0), _QuietHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    threedmol_stub = """
    window.$3Dmol = {
      Vector3: function(x, y, z) { return {x:x, y:y, z:z}; },
      createViewer: function() {
        return {
          models: [],
          addModel: function(text) { this.models.push({text:text}); },
          getModel: function() { return { selectedAtoms: function(){ return []; } }; },
          setStyle: function(){}, setHoverable: function(){}, setClickable: function(){},
          zoomTo: function(){}, render: function(){},
          addSphere: function(){ return {}; }, addCylinder: function(){ return {}; },
          addCustom: function(){ return {}; }, removeShape: function(){}
        };
      }
    };
    """
    rdkit_stub = """
    window.initRDKitModule = function() {
      return Promise.resolve({
        version: function(){ return 'stub'; },
        get_mol: function(){ return { is_valid: function(){ return false; }, delete: function(){} }; }
      });
    };
    """

    try:
        with sync_playwright() as p:
            try:
                browser = p.chromium.launch()
            except Exception as exc:
                pytest.skip(f"Playwright browser not installed: {exc}")
            page = browser.new_page()
            page.route("https://3Dmol.org/build/3Dmol-min.js", lambda route: route.fulfill(status=200, body=threedmol_stub, content_type="application/javascript"))
            page.route("https://unpkg.com/@rdkit/rdkit/dist/RDKit_minimal.js", lambda route: route.fulfill(status=200, body=rdkit_stub, content_type="application/javascript"))
            page.goto(f"http://127.0.0.1:{server.server_address[1]}/index.html")
            page.wait_for_function("window.__viewer_state && window.__viewer_state.frames.length === 2")
            payload = page.locator("#scripps-viewer-input").text_content()
            assert json.loads(payload)["xyz_text"].startswith("2\nE -1.0")
            assert page.evaluate("typeof toggle_measurement_atom") == "function"
            assert page.evaluate("typeof add_angle_wedge") == "function"
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        os.chdir(old_cwd)
