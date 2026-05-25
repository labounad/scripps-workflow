"""SSH-driven HPC integration tests.

These tests reach over SSH to a real HPC login node, exercise the
workflow runtime env, hit the nmr-data database, and run small
end-to-end commands. The goal is to catch the class of regression
that mocked tests can't see: a broken interpreter path in the env
activation hook, a stale alembic migration, a renamed binary, a
moved central-tree root, etc.

Activation:

    export SCRIPPS_WORKFLOW_HPC_HOST="login00-adm"
    # (or "user@host" — anything ssh understands as a destination)
    python -m pytest -m hpc tests/test_hpc_integration.py -v

Without ``SCRIPPS_WORKFLOW_HPC_HOST`` set, every test in this file
auto-skips with the standard "HPC env unset" reason. Normal
``python -m pytest`` runs skip them silently.

Required HPC-side assumptions:

* ``~/.ssh/config`` (or default SSH agent / key) authenticates
  non-interactively. Tests use ``BatchMode=yes`` so a password
  prompt fails fast rather than hanging.
* ``micromamba`` initialization lives in the user's login shell
  (``.bash_profile`` → ``.bashrc``), so ``bash -lc 'micromamba
  activate ...'`` works. The standard micromamba install does this
  automatically.
* The workflow + nmrdb envs are installed at the canonical paths:

    /gpfs/group/shenvi/envs/workflow312
    /gpfs/group/shenvi/envs/nmrdb

  Override via SCRIPPS_WORKFLOW_HPC_WORKFLOW_ENV / _NMRDB_ENV if not.
* ``NMR_DATABASE_URL`` and ``NMR_HPC_DATA_ROOT`` are exported by the
  workflow312 env's activate.d hooks (see ``nmr-data/README.md``).
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Configuration — pulled from env at module load
# ---------------------------------------------------------------------------


HPC_HOST: str | None = os.environ.get("SCRIPPS_WORKFLOW_HPC_HOST")

WORKFLOW_ENV: str = os.environ.get(
    "SCRIPPS_WORKFLOW_HPC_WORKFLOW_ENV",
    "/gpfs/group/shenvi/envs/workflow312",
)
NMRDB_ENV: str = os.environ.get(
    "SCRIPPS_WORKFLOW_HPC_NMRDB_ENV",
    "/gpfs/group/shenvi/envs/nmrdb",
)
SCRIPPS_WORKFLOW_REPO: str = os.environ.get(
    "SCRIPPS_WORKFLOW_HPC_REPO",
    "/gpfs/group/shenvi/code/scripps-workflow",
)
NMR_DATA_REPO: str = os.environ.get(
    "SCRIPPS_WORKFLOW_HPC_NMR_DATA_REPO",
    "/gpfs/group/shenvi/code/nmr-data",
)


#: Apply to every test class. Skips the whole file if the HPC host
#: env var isn't set, and tags everything ``hpc`` so ``pytest -m hpc``
#: targets just this file.
pytestmark = [
    pytest.mark.hpc,
    pytest.mark.skipif(
        not HPC_HOST,
        reason=(
            "SCRIPPS_WORKFLOW_HPC_HOST not set — HPC integration tests "
            "skipped. Set it (e.g. export SCRIPPS_WORKFLOW_HPC_HOST=login00-adm) "
            "to opt in."
        ),
    ),
]


# ---------------------------------------------------------------------------
# SSH command helpers
# ---------------------------------------------------------------------------


def _ssh(cmd: str, *, timeout: int = 60) -> subprocess.CompletedProcess:
    """Run a raw shell command on the HPC via SSH.

    ``BatchMode=yes`` fails fast on password prompts rather than
    hanging. ``ConnectTimeout=10`` bounds the initial handshake.

    Raises:
        CalledProcessError: non-zero exit on the remote.
        TimeoutExpired:     command exceeded ``timeout`` seconds.
    """
    assert HPC_HOST, "SSH called without SCRIPPS_WORKFLOW_HPC_HOST"
    return subprocess.run(
        [
            "ssh",
            "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=10",
            HPC_HOST,
            cmd,
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=True,
    )


def _ssh_in_env(
    cmd: str,
    *,
    env_path: str = WORKFLOW_ENV,
    timeout: int = 60,
) -> subprocess.CompletedProcess:
    """Run ``cmd`` inside an activated micromamba env on the HPC.

    Uses ``bash -lc`` so the user's login shell init sources the
    micromamba shell hook before ``activate`` runs. ``cmd`` may
    contain shell metacharacters — it's interpreted by bash on the
    remote side.
    """
    inner = f"micromamba activate {shlex.quote(env_path)} && {cmd}"
    return _ssh(f"bash -lc {shlex.quote(inner)}", timeout=timeout)


# ---------------------------------------------------------------------------
# Smoke tests — fast checks that the HPC env is healthy
# ---------------------------------------------------------------------------


class TestSshReachable:
    def test_ssh_handshake(self):
        """SSH connects + can run a trivial command."""
        result = _ssh("echo ready", timeout=15)
        assert result.stdout.strip() == "ready"

    def test_hostname_resolves(self):
        """``hostname`` returns a non-empty string."""
        result = _ssh("hostname", timeout=15)
        assert result.stdout.strip(), "remote hostname was empty"


class TestWorkflowEnv:
    """The workflow312 micromamba env activates and exposes the
    expected interpreter + entry-point binaries."""

    def test_env_dir_exists(self):
        _ssh(f"test -d {shlex.quote(WORKFLOW_ENV)}", timeout=15)

    def test_python_resolves_into_env(self):
        result = _ssh_in_env("which python")
        path = result.stdout.strip()
        assert path.startswith(WORKFLOW_ENV + "/bin/"), (
            f"python resolved to {path!r}, expected something under "
            f"{WORKFLOW_ENV}/bin/. The env may not be activating "
            f"cleanly via micromamba shell hook."
        )

    def test_scripps_workflow_importable(self):
        result = _ssh_in_env(
            "python -c 'import scripps_workflow; print(scripps_workflow.__file__)'"
        )
        assert "scripps_workflow/__init__.py" in result.stdout

    def test_nmr_data_importable(self):
        """The db extra is installed: nmr_data + sqlalchemy reachable."""
        result = _ssh_in_env(
            "python -c 'import nmr_data, sqlalchemy; print(nmr_data.__file__)'"
        )
        assert "nmr_data/__init__.py" in result.stdout

    def test_wf_node_entrypoints_on_path(self):
        """The console-script entry points are on PATH."""
        result = _ssh_in_env("which wf-embed wf-xtb wf-crest wf-db-ingest")
        # All four should resolve.
        lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
        assert len(lines) == 4, f"expected 4 entry points, got: {result.stdout}"


class TestEnvVars:
    """The workflow312 env's activate.d hooks export the database +
    HPC-data root env vars that the registry and cache layers need."""

    def test_NMR_DATABASE_URL_exported(self):
        result = _ssh_in_env('echo "$NMR_DATABASE_URL"')
        url = result.stdout.strip()
        assert url, (
            "NMR_DATABASE_URL is not exported after activating workflow312. "
            "Check the env's activate.d hook (see nmr-data/README.md HPC section)."
        )
        assert url.startswith(("postgresql://", "postgres://")), (
            f"NMR_DATABASE_URL doesn't look like a postgres URL: {url!r}"
        )

    def test_NMR_HPC_DATA_ROOT_exported(self):
        result = _ssh_in_env('echo "$NMR_HPC_DATA_ROOT"')
        root = result.stdout.strip()
        assert root, "NMR_HPC_DATA_ROOT is not exported after activating workflow312"
        # Sanity check: the path should be absolute.
        assert root.startswith("/"), f"NMR_HPC_DATA_ROOT not absolute: {root!r}"

    def test_NMR_HPC_DATA_ROOT_dir_exists(self):
        """The central-tree root is an actual directory."""
        result = _ssh_in_env(
            'test -d "$NMR_HPC_DATA_ROOT" && echo OK'
        )
        assert result.stdout.strip() == "OK"


class TestDatabase:
    """The database is reachable and at the expected schema version."""

    def test_db_query_succeeds(self):
        """``SELECT 1`` round-trips."""
        result = _ssh_in_env(
            'psql "$NMR_DATABASE_URL" -At -c "SELECT 1"'
        )
        assert result.stdout.strip() == "1"

    def test_alembic_at_head(self):
        """``alembic current`` matches ``alembic heads`` — schema isn't
        behind a pending migration."""
        # nmrdb env owns the alembic binary + the migration repo.
        current = _ssh_in_env(
            f"cd {shlex.quote(NMR_DATA_REPO)} && alembic current",
            env_path=NMRDB_ENV,
        ).stdout
        heads = _ssh_in_env(
            f"cd {shlex.quote(NMR_DATA_REPO)} && alembic heads",
            env_path=NMRDB_ENV,
        ).stdout

        # alembic current emits e.g. "763299503a0f (head)" — extract
        # the revision id (first token of each non-blank line).
        def _revs(out: str) -> set[str]:
            return {
                ln.strip().split()[0]
                for ln in out.splitlines()
                if ln.strip() and not ln.startswith(("INFO", "WARN", "DEBUG"))
            }

        cur, hd = _revs(current), _revs(heads)
        assert cur == hd, (
            f"alembic schema not at head: current={cur}, heads={hd}. "
            f"Run `alembic upgrade head` in the nmrdb env."
        )


class TestInspectRegistryProgress:
    """The tools/inspect_registry_progress.py script runs against the
    deployed scripps-workflow repo without crashing."""

    def test_script_runs_for_unknown_molecule(self):
        """Inspecting a SMILES that has no DB state is a normal call —
        the script should print a header and exit 0. Uses an obscure
        SMILES very unlikely to exist."""
        # Use a small but unusual molecule: 1-aminoadamantane (memantine
        # precursor). InChIKey unlikely to be in the test DB. The
        # script's output shape branches on molecule-presence:
        #   * absent → "DB state: molecule row absent..." + early return
        #   * present → per-stage tables + final "Summary:" block
        # Either is a successful run; we just need ONE of the two
        # terminal markers to appear so we know the script reached
        # its happy path.
        result = _ssh_in_env(
            f"cd {shlex.quote(SCRIPPS_WORKFLOW_REPO)} && "
            f"python tools/inspect_registry_progress.py 'NC12CC3CC(C1)CC(C2)C3'",
            timeout=30,
        )
        # Header always prints.
        assert "SMILES:" in result.stdout
        assert "InChIKey:" in result.stdout
        # Either branch's terminal marker.
        assert (
            "molecule row absent" in result.stdout
            or "Summary:" in result.stdout
        ), result.stdout


class TestWfEmbed:
    """The cheapest possible producing node — ``wf-embed`` runs an
    RDKit ETKDG embed + MMFF cleanup, no SLURM, no DB. Exercises the
    full Node base-class machinery (argv parsing, manifest write,
    pointer emit) without depending on any compute resource.
    """

    def test_wf_embed_emits_valid_pointer(self, tmp_path):
        """Run ``wf-embed smiles=CO`` in a fresh remote tmpdir; the
        emitted stdout line is a wf.pointer.v1 JSON with ok=true."""
        # Build a unique remote work dir under /tmp so we don't pollute
        # the user's home. ``mktemp -d`` is portable across Linux.
        result = _ssh_in_env(
            "remote_tmp=$(mktemp -d) && "
            "cd \"$remote_tmp\" && "
            "wf-embed smiles=CO && "
            "echo \"---SEP---\" && "
            "cat outputs/manifest.json && "
            "rm -rf \"$remote_tmp\"",
            timeout=60,
        )
        stdout = result.stdout
        assert "---SEP---" in stdout, (
            f"wf-embed didn't reach the manifest dump step:\n{stdout}"
        )
        pointer_line, manifest_blob = stdout.split("---SEP---", 1)
        pointer_line = pointer_line.strip().splitlines()[-1]

        pointer = json.loads(pointer_line)
        assert pointer.get("schema") == "wf.pointer.v1"
        assert pointer.get("ok") is True
        assert pointer.get("manifest_path"), pointer

        manifest = json.loads(manifest_blob.strip())
        assert manifest.get("ok") is True
        assert manifest.get("step") == "smiles_to_3d"
        # Embed records the input SMILES + at least one xyz artifact.
        assert manifest["inputs"].get("smiles") == "CO"
        assert manifest["artifacts"].get("xyz"), "no xyz artifact in manifest"


# ---------------------------------------------------------------------------
# Real-pipeline helpers (shared between dry-run and SLURM-submit variants)
# ---------------------------------------------------------------------------


def _quoted_argv_tail(*args: str) -> str:
    """Quote each token for inclusion in a shell command line."""
    return " ".join(shlex.quote(a) for a in args)


def _run_pipeline_step(
    remote_workdir: str,
    step_dir: str,
    cmd_argv: list[str],
    *,
    timeout: int,
) -> tuple[dict, dict]:
    """Run one ``wf-*`` invocation in ``remote_workdir/step_dir``.

    ``cmd_argv`` is the full command starting with the entry-point
    name (``wf-embed``, ``wf-xtb``, …) and any tokens after argv[1]
    (the upstream pointer, when applicable). Returns
    ``(pointer_dict, manifest_dict)``. Both are parsed from the
    remote outputs of the run.

    Implementation:

      mkdir -p <workdir>/<step_dir>
      cd <workdir>/<step_dir>
      <cmd_argv>                            # writes outputs/manifest.json + stdout pointer
      echo "---SEP---"
      cat outputs/manifest.json

    The stdout-pointer-line and the dumped manifest get returned to
    pytest space for assertions.
    """
    full_step = f"{remote_workdir}/{step_dir}"
    inner = (
        f"mkdir -p {shlex.quote(full_step)} && "
        f"cd {shlex.quote(full_step)} && "
        f"{_quoted_argv_tail(*cmd_argv)} && "
        f"echo '---SEP---' && "
        f"cat outputs/manifest.json"
    )
    result = _ssh_in_env(inner, timeout=timeout)
    stdout = result.stdout
    assert "---SEP---" in stdout, (
        f"step {step_dir!r} didn't reach the manifest dump:\n"
        f"--- stdout ---\n{stdout}\n--- stderr ---\n{result.stderr}"
    )
    pointer_text, manifest_blob = stdout.split("---SEP---", 1)
    pointer = json.loads(pointer_text.strip().splitlines()[-1])
    manifest = json.loads(manifest_blob.strip())
    return pointer, manifest


def _make_remote_workdir() -> str:
    """Create a ``mktemp -d`` workdir on the remote, return its path.

    Doesn't auto-clean — pytest tests can leak these for inspection;
    they live under /tmp which the cluster's per-node cleanup handles
    on reboot. To clean explicitly, call ``_remove_remote_workdir``.
    """
    result = _ssh("mktemp -d", timeout=15)
    path = result.stdout.strip().splitlines()[-1]
    assert path.startswith("/"), f"unexpected mktemp output: {path!r}"
    return path


def _remove_remote_workdir(path: str) -> None:
    """Best-effort cleanup. Test code should call this in a finally
    block; failure is non-fatal."""
    try:
        _ssh(f"rm -rf {shlex.quote(path)}", timeout=30)
    except subprocess.CalledProcessError:
        pass


# ---------------------------------------------------------------------------
# Dry-run real-pipeline test — no SLURM, exercises the inline-compute
# stages of the NMR pipeline (embed → xtb → crest → prism) plus the
# producer-side registry write on CREST.
# ---------------------------------------------------------------------------


class TestRealPipelineDryRun:
    """Run the inline-compute portion of the NMR pipeline against the
    real HPC environment. No SLURM submission. Verifies:

      * Each manifest is well-formed and ``ok=true``.
      * Pointer-chaining works across nodes (each node consumes the
        previous node's pointer as argv[1]).
      * The CREST producing node successfully self-registers an
        ``ensemble`` row to nmr-data (``manifest.inputs.registry.ensemble.ok=true``).
      * The ensemble row's central-tree dir exists on disk.

    Total runtime ~1–2 min on the cluster login node: ``mktemp`` +
    four SSH round trips (embed ~1 s, xtb ~1 s, crest quick mode ~30 s
    on methanol, prism ~1 s) plus the registry write. No compute-node
    allocation is made.

    Uses methanol (``CO``) as the test molecule — small enough that
    CREST quick mode finishes in <1 min on a login node without
    being a bad cluster citizen.
    """

    SMILES = "CO"

    def test_chain_embed_through_prism(self):
        workdir = _make_remote_workdir()
        try:
            # 1. wf-embed — RDKit ETKDG, no upstream.
            embed_ptr, embed_m = _run_pipeline_step(
                workdir, "01_embed", ["wf-embed", f"smiles={self.SMILES}"],
                timeout=60,
            )
            assert embed_m["ok"] is True
            assert embed_m["step"] == "smiles_to_3d"
            assert embed_m["artifacts"].get("xyz"), embed_m

            # 2. wf-xtb — xTB single-point/opt on the embedded geometry.
            xtb_ptr, xtb_m = _run_pipeline_step(
                workdir, "02_xtb",
                ["wf-xtb", json.dumps(embed_ptr), "calculations=[\"optimize\"]"],
                timeout=120,
            )
            assert xtb_m["ok"] is True
            assert xtb_m["step"] == "xtb_calc"

            # 3. wf-crest — conformer search in quick mode. The slow
            # step of this chain; ~30 s for a small molecule. Bump
            # the timeout generously for headwind.
            crest_ptr, crest_m = _run_pipeline_step(
                workdir, "03_crest",
                ["wf-crest", json.dumps(xtb_ptr), "mode=quick"],
                timeout=300,
            )
            assert crest_m["ok"] is True, crest_m.get("failures")
            assert crest_m["step"] == "crest"
            assert crest_m["artifacts"].get("conformers"), crest_m

            # 4. wf-crest's registry stash — the ensemble row should
            # have landed in the DB. Skipped silently if NMR_DATABASE_URL
            # isn't reachable, so we assert on the ``ok`` field rather
            # than requiring ``status="created"`` (the row may already
            # exist from a prior run).
            reg = crest_m["inputs"].get("registry", {}).get("ensemble")
            assert reg is not None, (
                "crest didn't stash a registry.ensemble block — check that "
                "NMR_DATABASE_URL is set and nmr_data is importable on the cluster"
            )
            assert reg["ok"] is True, reg
            assert reg["status"] in {"created", "reused"}, reg
            assert reg.get("central_tree_path"), reg
            # Ensemble dir exists on disk.
            _ssh_in_env(
                f"test -d \"$NMR_HPC_DATA_ROOT/{reg['central_tree_path']}\"",
                timeout=15,
            )

            # 5. wf-prism — RMSD/MoI pruning of the CREST output. Cheap.
            prism_ptr, prism_m = _run_pipeline_step(
                workdir, "04_prism",
                ["wf-prism", json.dumps(crest_ptr)],
                timeout=120,
            )
            assert prism_m["ok"] is True, prism_m.get("failures")
            assert prism_m["step"] == "prism_screen"
            assert prism_m["artifacts"].get("conformers"), prism_m

        finally:
            _remove_remote_workdir(workdir)


# ---------------------------------------------------------------------------
# SLURM-submitting real-pipeline test — opt-in via env var
# ---------------------------------------------------------------------------


_HPC_COMPUTE_ENABLED = bool(os.environ.get("SCRIPPS_WORKFLOW_HPC_ALLOW_SUBMIT"))


@pytest.mark.hpc_compute
@pytest.mark.skipif(
    not (HPC_HOST and _HPC_COMPUTE_ENABLED),
    reason=(
        "SCRIPPS_WORKFLOW_HPC_HOST and SCRIPPS_WORKFLOW_HPC_ALLOW_SUBMIT must "
        "both be set to run SLURM-submitting integration tests. These submit "
        "real sbatch jobs and consume compute-node time; opt in explicitly."
    ),
)
class TestRealPipelineSlurm:
    """SLURM-submitting variant of the real-pipeline test.

    Runs the same inline chain (embed → xtb → crest → prism) and then
    submits a real ``wf-orca-dft-array`` SLURM array on a tiny molecule
    with a short walltime. Verifies the array completes, the
    optimized geometries land in the manifest, and the ``DftRun`` row
    appears in the DB with its central-tree dir populated.

    Gated by ``@pytest.mark.hpc_compute`` plus
    ``SCRIPPS_WORKFLOW_HPC_ALLOW_SUBMIT=1`` — the existing ``hpc``
    marker alone won't pick this up. To run:

        export SCRIPPS_WORKFLOW_HPC_HOST=garibaldihpc
        export SCRIPPS_WORKFLOW_HPC_ALLOW_SUBMIT=1
        python -m pytest -m hpc_compute -v tests/test_hpc_integration.py

    Total runtime depends on queue wait; budget 20–40 min wallclock.
    Each invocation submits ~1 sbatch task with --time=00:10:00 on a
    small molecule, so the cluster-time footprint per run is minimal.
    """

    SMILES = "CO"

    def test_dft_array_submits_and_completes(self):
        workdir = _make_remote_workdir()
        try:
            embed_ptr, _ = _run_pipeline_step(
                workdir, "01_embed", ["wf-embed", f"smiles={self.SMILES}"],
                timeout=60,
            )
            xtb_ptr, _ = _run_pipeline_step(
                workdir, "02_xtb",
                ["wf-xtb", json.dumps(embed_ptr), "calculations=[\"optimize\"]"],
                timeout=120,
            )
            crest_ptr, crest_m = _run_pipeline_step(
                workdir, "03_crest",
                ["wf-crest", json.dumps(xtb_ptr), "mode=quick"],
                timeout=300,
            )
            assert crest_m["ok"] is True
            prism_ptr, prism_m = _run_pipeline_step(
                workdir, "04_prism",
                ["wf-prism", json.dumps(crest_ptr), "max_kept=1"],
                timeout=120,
            )
            assert prism_m["ok"] is True

            # ---- DFT array with REAL SLURM submit ----
            # Short walltime + minimal nprocs so the cluster-time
            # footprint is small. monitor=true blocks until the array
            # finishes; SSH timeout is the safety net.
            dft_ptr, dft_m = _run_pipeline_step(
                workdir, "05_dft",
                [
                    "wf-orca-dft-array", json.dumps(prism_ptr),
                    "submit=true",
                    "monitor=true",
                    "time_limit=00:10:00",
                    "nprocs=4",
                    "maxcore=2000",
                    "monitor_interval_s=15",
                    "monitor_timeout_min=30",
                ],
                # SSH timeout = SLURM-time + queue-wait + slack.
                timeout=60 * 45,
            )
            assert dft_m["ok"] is True, dft_m.get("failures")
            assert dft_m["step"] == "orca_dft_array"
            # The DFT array must have produced at least one optimized
            # geometry in the conformers bucket.
            assert dft_m["artifacts"].get("conformers"), dft_m

            # Registry stash — DftRun row must have landed.
            reg = dft_m["inputs"].get("registry", {}).get("dft_run")
            assert reg is not None, (
                "orca_dft_array didn't stash a registry.dft_run block"
            )
            assert reg["ok"] is True, reg
            assert reg["status"] in {"created", "reused"}, reg
            assert reg.get("central_tree_path"), reg
            _ssh(
                f"test -d $NMR_HPC_DATA_ROOT/{reg['central_tree_path']}",
                timeout=15,
            )
        finally:
            _remove_remote_workdir(workdir)
