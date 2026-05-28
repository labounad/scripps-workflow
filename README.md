# scripps-workflow

A small, HPC-aware framework for chaining quantum-chemistry, cheminformatics, NMR prediction, and
data-ingest tools into reproducible pipelines.

## Status

**Pre-alpha.** APIs, wire formats, node contracts, and GUI integration are still moving. Pin a commit
if you depend on this.

The engine that schedules these workflows on SLURM is a separate hosted service. This repository
contains the framework code, node implementations, output viewers, GUI node exporters, and workflow
logic that the engine round-trips through. Every node is also a normal command-line program and can
be run or tested outside the GUI.

## What it is

Each process node is a small command-line program with a fixed wire protocol:

- reads **config** from `argv` (`key=value` tokens, or JSON for nested config)
- reads **upstream results** from `stdin` as a one-line JSON pointer to a manifest file
- writes **outputs** into a per-call `outputs/` directory
- emits its own one-line JSON pointer to `stdout`

Nodes compose with plain Unix pipes:

```bash
wf-embed smiles="CCO" | wf-xtb theory=gfn2 calculations='["optimize","sp_energy"]'
```

The pointer format is `wf.pointer.v1`:

```json
{"schema": "wf.pointer.v1", "ok": true, "manifest_path": "/abs/path/to/manifest.json"}
```

The manifest (`wf.result.v1`) records the node's inputs, outputs, environment, and per-artifact
metadata. A failed node can still emit a pointer with `ok: false` plus an error block in the
manifest, allowing downstream nodes to decide how to react. Hard-fail behavior is opt-in via
`fail_policy=hard`.

## Long-term NMR vision

`scripps-workflow` is the compute engine for a larger NMR data platform.

The workflow generates high-value computed NMR data:

```text
molecule
  → conformer search
  → DFT geometry optimization
  → thermochemistry
  → NMR shielding / coupling calculations
  → Boltzmann aggregation
  → structured database ingest
```

Together with the companion `nmr-data` repository, the long-term goal is to connect:

1. computed structure-to-spectrum data
2. experimental NMR data produced by institute instruments
3. future machine-learning and data-science workflows

This enables future work such as:

- machine-learned structure-to-spectrum prediction
- spectrum-to-structure ranking and stereochemical assignment
- automatic shift and J-coupling extraction
- comparison of benchtop spectra to simulated high-field spectra
- large-scale analysis of DFT/NMR error patterns by molecule class, method, solvent, and nucleus

The workflow therefore has two roles:

1. run expensive, reproducible quantum-chemical calculations on HPC resources
2. emit structured artifacts and metadata that can be organized by PostgreSQL into a reusable
   scientific dataset

## NMR prediction workflow

A representative computational NMR workflow is:

```text
wf-embed
  → wf-xtb
  → wf-crest / wf-orca-goat
  → wf-prism / wf-marc
  → wf-orca-dft-array
  → wf-orca-thermo-array
  → wf-thermo-aggregate
  → wf-nmr-aggregate
  → wf-db-ingest
```

Conceptually:

1. **Conformer generation and screening**
   - RDKit embedding, xTB optimization, CREST/GOAT search, PRISM/MARC pruning.
2. **DFT geometry optimization**
   - Slurm-array ORCA jobs, often one task per retained conformer.
3. **Thermochemistry**
   - frequency / thermochemistry jobs and high-level single points.
4. **NMR property calculations**
   - separate NMR jobs for proton shifts, carbon shifts, and J-couplings.
5. **Aggregation**
   - parse per-conformer outputs, compute Boltzmann weights, apply referencing/calibration, emit
     predicted shifts/couplings/XMLs/manifests.
6. **Database ingest**
   - copy/organize heavy artifacts under the central HPC data tree and insert metadata/scalars/paths
     into the `nmr-data` schema.

The computationally expensive work is in conformer search and ORCA arrays. The database traffic is
small and bursty: cache lookup at the beginning, registration/ingest at the end.

## Filesystem and database model

Large files remain on GPFS. PostgreSQL is an optional structured index/cache/provenance layer.

Typical group deployment:

```bash
export SCRIPPS_WORKFLOW_ROOT="/gpfs/group/shenvi/code/scripps-workflow"
export NMR_HPC_DATA_ROOT="/gpfs/group/shenvi/nmr-data/runs"
export NMR_DATABASE_URL="postgresql://nmrdata:<password>@<db-host>:5432/nmrdata"
```

Important rule:

- Use a real service hostname for PostgreSQL when running from Slurm.
- Do not rely on `localhost` unless the database server is running on the same node as the job.

If the database is unavailable, cache-aware compute nodes should treat this as a **cache miss** and
continue. Database availability improves reuse and provenance; it should not be a correctness
dependency for running the calculation.

## Repository layout

```text
src/scripps_workflow/
    node.py             # Node base class + lifecycle + _try_register_to_nmr_data hook
    schema.py           # Manifest / artifact / pointer dataclasses
    pointer.py          # Pointer read/write helpers
    nmr_calibration.py  # Linear-scaling table (cheshire + lab-fit) for shifts + J's
    equivalence.py      # Three-tier chemical-equivalence detector (HARD/SOFT/NONE)
    contracts/          # Cross-node contracts
    nodes/              # Concrete process-node implementations
    output_viewers/     # Standalone viewer bundle builders and static assets
    tag.py              # Tag-node wiring shims
tools/
    export_nodes.py                     # process-node exporter
    gen_output_node_ensemble_viewer.py  # GUI output-node bundle generator
    gen_output_node_geometry_viewer.py  # GUI output-node bundle generator
    output_node_bundle.py               # shared output-node generator helpers
    gui_export_config.py                # shared GUI/HPC path config
    inspect_registry_progress.py        # ad-hoc DB + central-tree inspector
    check_group_portability.sh          # group-paths-vs-user-paths audit
tests/
    pytest suite
    test_hpc_integration.py             # SSH-driven HPC smoke tests (opt-in)
docs/
    refactor-roadmap.md                 # ongoing software-structure roadmap
    registry-design.md                  # per-stage self-registration design
    registry-verification.md            # 4-claim runbook for the registry
    hpc-database-setup.md               # PostgreSQL deployment notes
    CONFIG_REFERENCE.md                 # auto-generated per-node config reference
```

## Install for development

Editable install with test extras:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
```

Optional extras gate heavier dependencies:

| Extra   | Pulls in                             | Used by                                      |
|---------|--------------------------------------|----------------------------------------------|
| `test`  | pytest, pyyaml                       | test suite                                    |
| `chem`  | numpy, rdkit                         | cheminformatics / QC-related nodes           |
| `gui`   | pydantic, pyyaml                     | GUI export/import round-trip tooling         |
| `prism` | prism-pruner, Python 3.12 ecosystem  | PRISM conformer-pruning node                  |
| `all`   | everything                           | development                                  |

Prefer:

```bash
python -m pytest
```

rather than bare `pytest`, so the test runner uses the same interpreter/environment as `python`.

## HPC group deployment

Canonical group paths:

```text
/gpfs/group/shenvi/code/scripps-workflow
/gpfs/group/shenvi/code/nmr-data
/gpfs/group/shenvi/envs/workflow312
/gpfs/group/shenvi/software/xtb-crest
/gpfs/group/shenvi/nmr-data/runs
```

Install from the group repository:

```bash
cd /gpfs/group/shenvi/code/scripps-workflow
/gpfs/group/shenvi/envs/workflow312/bin/python -m pip install -e .
```

Check portability:

```bash
bash tools/check_group_portability.sh
```

Expected properties:

- `scripps_workflow` imports from `/gpfs/group/shenvi/code/scripps-workflow`
- `xtb` and `crest` resolve from `/gpfs/group/shenvi/software/xtb-crest/bin`
- `NMR_HPC_DATA_ROOT` points to `/gpfs/group/shenvi/nmr-data/runs`
- no runtime paths point into `/gpfs/home/labounader` or `/gpfs/group/shenvi/Users/labounader`

## Running tests

```bash
python -m pytest -q
```

Single-file examples:

```bash
python -m pytest -q tests/test_xtb_calc.py
python -m pytest -q tests/test_output_viewer_bundles.py
```

Browser smoke tests for the standalone viewers use Playwright if installed:

```bash
python -m pip install pytest-playwright
python -m playwright install chromium
python -m pytest -q tests/test_output_viewer_browser_smoke.py
```

SSH-driven HPC integration tests live in `tests/test_hpc_integration.py` and are gated by both
the `hpc` pytest marker and a `SCRIPPS_WORKFLOW_HPC_HOST` env var. Default `pytest` runs skip
them silently. To run against a real cluster:

```bash
export SCRIPPS_WORKFLOW_HPC_HOST=garibaldihpc      # or whatever ~/.ssh/config alias
python -m pytest -m hpc -v tests/test_hpc_integration.py
```

These probe SSH reachability, the workflow312 micromamba env, `NMR_DATABASE_URL` /
`NMR_HPC_DATA_ROOT` activation hooks, alembic schema head, `inspect_registry_progress.py`, and a
`wf-embed` round-trip. The opt-in flag is `pytest.mark.hpc`.

## Available nodes

| Entry point             | Purpose                                                       |
|-------------------------|---------------------------------------------------------------|
| `wf-embed`              | SMILES → 3D coordinates                                       |
| `wf-xtb`                | xTB single-point / optimize / gradient / hessian              |
| `wf-crest`              | CREST conformer search                                        |
| `wf-orca-goat`          | ORCA GOAT global conformer search                             |
| `wf-prism`              | RMSD/MoI conformer pruning via prism-pruner                   |
| `wf-marc`               | navicat-marc clustering                                       |
| `wf-orca-dft-array`     | Slurm-array DFT geometry optimization                         |
| `wf-orca-thermo-array`  | Composite thermo + high-level SP + optional NMR jobs          |
| `wf-thermo-aggregate`   | Boltzmann weights and composite Gibbs energies                |
| `wf-nmr-aggregate`      | Boltzmann-averaged NMR shifts/couplings and calibration       |
| `wf-db-ingest`          | Ingest final NMR aggregate products into `nmr-data`           |
| `wf-tag-input`          | GUI wiring shim for key/value relay                           |
| `wf-artifact-export`    | Export selected artifacts into concrete files                 |
| `wf-extract-conformers` | Extract conformers from upstream pointer/manifest artifacts   |

## `wf-orca-thermo-array` NMR protocol

`wf-orca-thermo-array` runs a per-conformer compound protocol inside a Slurm array task. Each task
invokes ORCA multiple times sequentially:

- `orca_thermo.inp` — r2scan-3c frequency / thermochemistry
- `orca_thermo_sp.inp` — wB97M-V/def2-TZVPP high-level single point, output concatenated into the
  thermo `.out` after completion. Run as a separate ORCA process to avoid `$new_job` method-state
  leakage between the two functionals.
- `orca_nmr_h.inp` — proton shieldings (WP04 / 6-311++G(2d,p) by default)
- `orca_nmr_c.inp` — carbon shieldings (wB97X-D / 6-31G(d,p) by default)
- `orca_nmr_j.inp` — J-couplings (mPW1PW91 / pcJ-2 by default)

The NMR calculations are separate ORCA processes so method-state flags, dispersion settings,
nonlocal correlation flags, and basis choices do not leak between chemically unrelated functionals.

The node emits raw ORCA outputs. `wf-nmr-aggregate` handles parsing, Boltzmann population weighting,
and linear-scaling calibration.

### `system_class` profile (organopalladium and other heavy-metal NMR)

A `system_class` config knob switches the NMR-job recipe based on the chemistry of the input:

| Profile     | When it fires                                  | What changes                                                                                   |
|-------------|------------------------------------------------|------------------------------------------------------------------------------------------------|
| `organic`   | Default for molecules with no Tier 2 metal     | WP04 / wB97X-D / mPW1PW91 with the organic basis sets. No relativistic Hamiltonian             |
| `organopd`  | Auto-selected when `Pd` is in input            | Same functionals, but ORCA's `! ZORA` Hamiltonian + def2-ZORA-TZVPP basis on all three NMR jobs |
| `organopt` / `organorh` / `organoir` / `organoau` / `organohg` / `organoru` / `organoag` / `organomo` / `organore` / `organoos` / `organow` | Auto-selected per metal (Pt, Rh, Ir, Au, Hg, ...) | Same ZORA + def2-ZORA-TZVPP treatment as `organopd`; distinct profile names so future per-metal calibration tables (TODO #89) can fork |
| `organotm`  | Generic transition-metal umbrella — fallback for Tier 2 elements without a dedicated profile (Tc, Cd, lanthanides) | Same ZORA + def2-ZORA-TZVPP treatment |

Geometry opt + freq + high-level SP are unchanged across profiles — def2 ECPs already handle the
scalar relativistic contraction. The ZORA path is only needed for NMR shieldings, where the HALA
effect on ¹H/¹³C near a heavy metal is missing without a relativistic Hamiltonian (typically
~0.5–2 ppm on ¹H, ~5–20 ppm on ¹³C, well above noise).

`system_class=auto` (the default) reads the upstream conformer xyz and picks the matching profile;
explicit settings (`system_class=organic`, `organopd`, `organopt`, ...) force the choice. The cache
fingerprint for `PredictedRun` naturally diverges between profiles because the basis-set fields are
part of the fingerprint payload, so ZORA and non-ZORA runs don't collide.

### Tier 1 heavy-atom basis supplementation (Br, I, Se, Sn, ...)

Light-heavy elements that don't need full ZORA but DO need a basis the configured set doesn't cover
(Br outside `pcJ-2`'s organic-main-group set, I outside Pople through Kr, Sn outside `6-311++G(2d,p)`'s
diffuse augmentation, etc.) are auto-supplemented per-element using ORCA's `%basis newgto` block.
The calibrated light-atom basis stays put; only the heavy atom gets the override.

Two knobs control the behavior:

- `heavy_atom_basis` (default `def2-TZVPP`) — the basis attached to each uncovered Tier 1 element.
  `def2-TZVPP` spans Z=1..86 with built-in Stuttgart ECPs for Z≥37, matching the high-level SP recipe.
- `on_uncovered_heavy_metal` (default `fail`) — policy when a Tier 2 metal slips through despite an
  explicit `system_class=organic`. `fail` raises immediately. `warn` logs and continues (likely to
  abort at ORCA read time). `auto_switch_profile` promotes the system_class to the matching Tier 2
  profile (e.g. `organopd` for Pd).

The supplemented basis is recorded into the cached basis identity as e.g.
`6-31G(d,p)+def2-TZVPP/heavy`, so a re-run with a different `heavy_atom_basis` correctly invalidates.
The element coverage tables live in `src/scripps_workflow/basis_coverage.py`; extend
`BASIS_ELEMENT_COVERAGE` when adding a new named basis to the defaults.

### Heteronuclear J auto-detection (¹⁹F / ³¹P)

An `auto_heteronuclear` config knob (default `true`) extends the J-coupling job to include
heteronuclear partners present in the molecule:

- ¹⁹F detected → `"all F"` appended to `coupling_pairs`, so ORCA computes H–F / C–F / F–F J's.
- ³¹P detected → `"all P"` appended, so H–P / C–P / P–P J's are computed.

`wf-nmr-aggregate` mirrors the same detection on its own upstream chain and auto-expands
`mnova_heteronuclear_partners`, so the simulated ¹H / ¹³C spectra render with realistic
heteronuclear splitting rather than the {¹⁹F} or {³¹P} decoupled view. Set `auto_heteronuclear=false`
on both nodes to force the decoupled spectrum.

Other heavy nuclei (Pd, quadrupolar metals, etc.) are deliberately excluded — their J-coupling
treatment requires a different methodology than mPW1PW91/pcJ-2.

### Equivalence-grouped CSV output

`wf-nmr-aggregate`'s `predicted_shifts.csv` and `predicted_couplings.csv` carry both per-atom rows
(unchanged from earlier) and equivalence-group annotation columns: `group_label`, `group_tier`
(`hard`/`soft`/`none`), `group_n_atoms`, `group_atom_indices`, and the group-averaged σ/δ values.
Collapse the CSV by `group_label` in pandas / Excel for a spectrum-comparison view (one row per
peak); use the atom-level columns for atom-by-atom analysis.

When no SMILES is available and the conformer xyz can't be perceived by RDKit, each atom lands in
its own NONE-tier singleton group labelled `atom_<idx>`. The CSV shape is invariant.

### Calibration

The default ¹³C linear-scaling calibration (`wB97X-D/6-31G(d,p)/CHCl3`) is a lab-fit recalibration
of cheshire against organopalladium experimental data (slope `−0.9781`, intercept `197.67`,
R²=0.9986, RMSE=1.51 ppm on the fit set). Reset to cheshire's original `−1.0501, 187.25` via an
explicit `solvent=CHCl3_CHESHIRE_2006` override on the calibration lookup. ¹H calibration retains
the published cheshire WP04 values. See `src/scripps_workflow/nmr_calibration.py` for the full
table and provenance.

## Standalone output viewers

This repository also builds standalone downloadable viewer bundles for GUI output nodes:

- ensemble viewer
- single-geometry viewer

These viewers are script-backed Layout nodes. The GUI node shim calls into the installed
`scripps_workflow` package, which builds a local ZIP containing `index.html`, static JS/CSS assets,
embedded payload metadata, and molecule/conformer data.

The viewer bundle is intended to be downloaded from the workflow run and opened locally in a browser.

## Cache and database behavior

Compute nodes both **read from** and **write to** the `nmr-data` database. Reads short-circuit
expensive deterministic stages; writes populate the central tree as each stage finishes, rather
than waiting for the terminal `wf-db-ingest` to ingest the whole pipeline at once.

### Read side — cache lookup

Cache-aware nodes consult the `nmr-data` database at the top of `run()` to see whether an
expensive deterministic result already exists:

- cache hit: emit cached manifest / reuse existing artifacts, skip the SLURM array entirely
- cache miss: compute normally
- database unavailable: warn and compute normally (fail-open)

This prevents temporary PostgreSQL outages from breaking calculations.

### Write side — per-stage self-registration

Each producing node calls `Node._try_register_to_nmr_data(stage, ...)` after a successful
compute. The base-class helper delegates to the matching `nmr_data.registry.register_<stage>`
function, which writes the row + copies artifacts into the central tree under
`NMR_HPC_DATA_ROOT/<inchikey>/<stage>/<uuid>/`. The four stages:

| Node                 | Stage          | Row table              | Central-tree dir                 |
|----------------------|----------------|------------------------|----------------------------------|
| `wf-crest`           | `ensemble`     | `conformer_ensembles`  | `ensembles/<uuid>/conformers/`   |
| `wf-orca-goat`       | `ensemble`     | `conformer_ensembles`  | `ensembles/<uuid>/conformers/`   |
| `wf-orca-dft-array`  | `dft_run`      | `dft_runs`             | `dft_runs/<uuid>/conformers/`    |
| `wf-orca-thermo-array` | `thermo_run` | `thermo_runs`          | `thermo_runs/<uuid>/conformers/` |
| `wf-nmr-aggregate`   | `predicted_run`| `predicted_runs`       | `predicted_runs/<uuid>/`         |

Registration is fail-open, mirroring the cache read side: a degraded DB never breaks a running
pipeline, the manifest just records `registry.<stage>.ok = false` for visibility.

`wf-db-ingest` survives as a **backstop**: a fast no-op when the producing nodes already
registered everything; a populator-of-last-resort if any stage's self-registration was skipped
(transient DB outage, missing env, etc.). Its `[INFO]` lines distinguish "backstop wrote nothing
— pipeline was healthy" from "backstop had to write — check why upstream skipped".

See `docs/registry-design.md` for the rationale and `docs/registry-verification.md` for the
end-to-end checklist + `tools/inspect_registry_progress.py` (a helper that prints the DB +
central-tree state for a given SMILES).

## GUI node export

Process nodes and output nodes are exported into GUI-importable ZIP bundles using:

```bash
python tools/export_nodes.py
python tools/gen_output_node_ensemble_viewer.py
python tools/gen_output_node_geometry_viewer.py
```

Output-viewer node shims bake in group-readable fallback source paths from `tools/gui_export_config.py`.
After changing GUI-facing node definitions or shim templates, regenerate and reimport the affected
node ZIPs.

## License

MIT. See `pyproject.toml`.

## Author

Lucas Abounader, The Scripps Research Institute.
