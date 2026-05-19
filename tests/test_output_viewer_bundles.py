import importlib.util
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

from scripps_workflow.output_viewers.artifact_resolver import resolve_ensemble_source, resolve_geometry_source
from scripps_workflow.pointer import Pointer

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_XYZ = ROOT / "tests" / "fixtures" / "crest_ensemble_33conf.xyz"


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _write_manifest(path: Path, *, cwd: Path, artifacts: dict, smiles: str = "CCO") -> None:
    base = {
        "schema": "wf.result.v1",
        "ok": True,
        "step": "upstream",
        "created_at_unix": 0,
        "runtime_seconds": 0.0,
        "cwd": str(cwd),
        "inputs": {"smiles": smiles},
        "environment": {},
        "upstream": {},
        "artifacts": {
            "logs": [],
            "xyz": [],
            "xyz_ensemble": [],
            "accepted": [],
            "rejected": [],
            "selected": [],
            "conformers": [],
            "files": [],
            "array": {},
        },
        "failures": [],
    }
    base["artifacts"].update(artifacts)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(base), encoding="utf-8")


def _assert_layout_bundle(zip_path: Path, node_name: str, output_name: str):
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        manifest = json.loads(zf.read(f"{node_name}.json"))
        script_path = manifest["files_info"][0]["path"]
        assert f"{node_name}.json" in names
        assert script_path in names
        assert manifest["node_type"] == "Output"
        assert manifest["category"] == "Layout"
        assert manifest["inputs"][0]["name"] == "source"
        assert manifest["inputs"][0]["type"] == "text"
        block = manifest["layout"]["Blocks"][0]
        assert block["output_file_name"] == output_name
        assert block["output_file_type"] == "zip"
        assert block["Files"][0]["path"] == script_path
    return script_path


def _run_output_script(node_zip: Path, script_path: str, tmp_path: Path, source: str, output_name: str):
    with zipfile.ZipFile(node_zip) as zf:
        zf.extract(script_path, path=tmp_path)
    script = tmp_path / script_path
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(
        [sys.executable, str(script), source],
        cwd=tmp_path,
        check=True,
        env=env,
    )
    out = tmp_path / output_name
    assert out.is_file()
    with zipfile.ZipFile(out) as zf:
        names = set(zf.namelist())
        assert "index.html" in names
        assert "js/viewer.js" in names
        assert "css/styles.css" in names
        assert "data/ensemble.xyz" in names or "data/geometry.xyz" in names
        assert "viewer_input.json" in names
        html = zf.read("index.html").decode("utf-8")
        assert "viewer-input" in html
        assert "xyz_text" in html
    return out


def test_resolver_reads_pointer_json_to_ensemble_and_geometry(tmp_path):
    xyz = tmp_path / "accepted_ensemble.xyz"
    xyz.write_text(
        "2\nE -1.0\nH 0 0 0\nH 0 0 1\n"
        "2\nE -0.9\nH 0 0 0\nH 0 0 2\n",
        encoding="utf-8",
    )
    mf = tmp_path / "outputs" / "manifest.json"
    _write_manifest(
        mf,
        cwd=tmp_path,
        artifacts={"xyz_ensemble": [{"label": "accepted_ensemble", "path_abs": str(xyz), "format": "xyz"}]},
    )
    ptr = Pointer.of(ok=True, manifest_path=mf).to_json_line()

    ens = resolve_ensemble_source(ptr)
    assert ens.n_frames == 2
    assert ens.smiles == "CCO"

    geom = resolve_geometry_source(ptr, conformer_index="2")
    assert geom.n_frames == 1
    assert "0 0 2" in geom.xyz_text


def test_ensemble_viewer_generates_standalone_zip_bundle_from_pointer(tmp_path):
    xyz = tmp_path / "accepted_ensemble.xyz"
    xyz.write_text(FIXTURE_XYZ.read_text(encoding="utf-8"), encoding="utf-8")
    mf = tmp_path / "outputs" / "manifest.json"
    _write_manifest(
        mf,
        cwd=tmp_path,
        artifacts={"xyz_ensemble": [{"label": "accepted_ensemble", "path_abs": str(xyz), "format": "xyz"}]},
    )
    ptr = Pointer.of(ok=True, manifest_path=mf).to_json_line()

    gen = _load_module(ROOT / "tools" / "gen_output_node_ensemble_viewer.py")
    node_zip = gen.write_zip(tmp_path)
    script_path = _assert_layout_bundle(
        node_zip,
        "ensemble_viewer",
        "ensemble_viewer_bundle.zip",
    )
    out = _run_output_script(node_zip, script_path, tmp_path, ptr, "ensemble_viewer_bundle.zip")
    with zipfile.ZipFile(out) as zf:
        html = zf.read("index.html").decode("utf-8")
        js = zf.read("js/viewer.js").decode("utf-8")
        css = zf.read("css/styles.css").decode("utf-8")
        assert "structure-2d" in html
        assert "RDKit_minimal.js" in html
        assert "toggle_structure_expanded" in js
        assert "align_to_selection" in js
        assert "reset_alignment" in js
        assert "#structure-2d" in css


def test_geometry_viewer_generates_standalone_zip_bundle_from_pointer(tmp_path):
    xyz = tmp_path / "accepted_ensemble.xyz"
    xyz.write_text(FIXTURE_XYZ.read_text(encoding="utf-8"), encoding="utf-8")
    mf = tmp_path / "outputs" / "manifest.json"
    _write_manifest(
        mf,
        cwd=tmp_path,
        artifacts={"xyz_ensemble": [{"label": "accepted_ensemble", "path_abs": str(xyz), "format": "xyz"}]},
    )
    ptr = Pointer.of(ok=True, manifest_path=mf).to_json_line()

    gen = _load_module(ROOT / "tools" / "gen_output_node_geometry_viewer.py")
    node_zip = gen.write_zip(tmp_path)
    script_path = _assert_layout_bundle(
        node_zip,
        "geometry_viewer",
        "geometry_viewer_bundle.zip",
    )
    _run_output_script(node_zip, script_path, tmp_path, ptr, "geometry_viewer_bundle.zip")


def test_output_viewer_assets_are_static_files():
    from scripps_workflow.output_viewers import assets

    static_root = ROOT / "src" / "scripps_workflow" / "output_viewers" / "static"
    ensemble_js_files = sorted((static_root / "ensemble" / "js").glob("*.js"))

    assert (static_root / "ensemble" / "index.html").is_file()
    assert (static_root / "geometry" / "index.html").is_file()
    assert (static_root / "geometry" / "viewer.js").is_file()
    assert (static_root / "common" / "styles.css").is_file()
    assert ensemble_js_files

    joined = "\n\n".join(p.read_text(encoding="utf-8").rstrip() for p in ensemble_js_files) + "\n"
    assert assets.ENSEMBLE_VIEWER_JS == joined
    assert "toggle_measurement_atom" in assets.ENSEMBLE_VIEWER_JS
    assert "add_angle_wedge" in assets.ENSEMBLE_VIEWER_JS


def test_output_viewer_generators_use_shared_layout_helper():
    ensemble_gen = _load_module(ROOT / "tools" / "gen_output_node_ensemble_viewer.py")
    geometry_gen = _load_module(ROOT / "tools" / "gen_output_node_geometry_viewer.py")
    helper = (ROOT / "tools" / "output_node_bundle.py").read_text(encoding="utf-8")

    assert ensemble_gen.SPEC.entrypoint == "scripps_workflow.output_viewers.ensemble_bundle:main"
    assert geometry_gen.SPEC.entrypoint == "scripps_workflow.output_viewers.geometry_bundle:main"
    assert "DEFAULT_WORKFLOW_PYTHON" in (ROOT / "tools" / "gui_export_config.py").read_text(encoding="utf-8")
    assert "def render_script_py" in helper
    assert "SCRIPPS_VIEWER_OUTPUT_DIR" in helper


def test_measurement_code_is_split_from_3dmol_rendering():
    js_root = ROOT / "src" / "scripps_workflow" / "output_viewers" / "static" / "ensemble" / "js"
    math_js = (js_root / "91_measurement_math.js").read_text(encoding="utf-8")
    shapes_js = (js_root / "94_measurement_shapes.js").read_text(encoding="utf-8")
    render_js = (js_root / "99_3dmol_rendering.js").read_text(encoding="utf-8")

    assert "function dihedral_abcd" in math_js
    assert "function angle_abc" in math_js
    assert "addCylinder" not in math_js
    assert "function add_angle_wedge" in shapes_js
    assert "function repaint_viewer" in render_js
