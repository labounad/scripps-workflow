import json
from pathlib import Path

from scripps_workflow.nodes import extract_conformers
from scripps_workflow.pointer import Pointer


def _write_manifest(path: Path, *, cwd: Path, artifacts: dict) -> None:
    base = {
        "schema": "wf.result.v1",
        "ok": True,
        "step": "upstream",
        "created_at_unix": 0,
        "runtime_seconds": 0.0,
        "cwd": str(cwd),
        "inputs": {},
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


def _xyz(path: Path, z: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"2\nE {-1.0 + z / 10:.6f}\nH 0 0 0\nH 0 0 {z}\n",
        encoding="utf-8",
    )


def test_extract_conformers_defaults_to_multiframe_from_records(tmp_path, monkeypatch, capsys):
    upstream = tmp_path / "upstream"
    conf1 = upstream / "conf_0001.xyz"
    conf2 = upstream / "conf_0002.xyz"
    _xyz(conf1, 1)
    _xyz(conf2, 2)
    manifest_path = upstream / "outputs" / "manifest.json"
    _write_manifest(
        manifest_path,
        cwd=upstream,
        artifacts={
            "conformers": [
                {"index": 1, "label": "conf_0001", "path_abs": str(conf1), "format": "xyz"},
                {"index": 2, "label": "conf_0002", "path_abs": str(conf2), "format": "xyz"},
            ]
        },
    )
    pointer = Pointer.of(ok=True, manifest_path=manifest_path).to_json_line()

    call = tmp_path / "call"
    call.mkdir()
    monkeypatch.chdir(call)
    rc = extract_conformers._run(["wf-extract-conformers", pointer])

    assert rc == 0
    stdout_path = Path(capsys.readouterr().out.strip())
    assert stdout_path.name == "extracted_conformers.xyz"
    text = stdout_path.read_text(encoding="utf-8")
    assert text.count("\n2\nE") == 1  # second frame boundary
    assert text.startswith("2\nE")
    manifest = json.loads((call / "outputs" / "manifest.json").read_text())
    assert manifest["ok"] is True
    assert manifest["inputs"]["resolved_mode"] == "all"
    assert manifest["inputs"]["n_extracted"] == 2
    assert manifest["artifacts"]["xyz_ensemble"][0]["path_abs"] == str(stdout_path)


def test_extract_conformers_single_frame_from_xyz_ensemble(tmp_path, monkeypatch, capsys):
    upstream = tmp_path / "upstream"
    ensemble = upstream / "accepted_ensemble.xyz"
    ensemble.parent.mkdir(parents=True, exist_ok=True)
    ensemble.write_text(
        "2\nE -1.0\nH 0 0 0\nH 0 0 1\n"
        "2\nE -0.9\nH 0 0 0\nH 0 0 2\n",
        encoding="utf-8",
    )
    manifest_path = upstream / "outputs" / "manifest.json"
    _write_manifest(
        manifest_path,
        cwd=upstream,
        artifacts={
            "xyz_ensemble": [
                {"label": "accepted_ensemble", "path_abs": str(ensemble), "format": "xyz"}
            ]
        },
    )
    pointer = Pointer.of(ok=True, manifest_path=manifest_path).to_json_line()

    call = tmp_path / "call"
    call.mkdir()
    monkeypatch.chdir(call)
    rc = extract_conformers._run([
        "wf-extract-conformers",
        pointer,
        "conformer_index=2",
    ])

    assert rc == 0
    stdout_path = Path(capsys.readouterr().out.strip())
    assert stdout_path.name == "conformer_0002.xyz"
    assert stdout_path.read_text(encoding="utf-8") == "2\nE -0.9\nH 0 0 0\nH 0 0 2\n"
    manifest = json.loads((call / "outputs" / "manifest.json").read_text())
    assert manifest["artifacts"]["xyz"][0]["path_abs"] == str(stdout_path)
