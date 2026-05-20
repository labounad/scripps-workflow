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
    node.py             # Node base class + lifecycle
    schema.py           # Manifest / artifact / pointer dataclasses
    pointer.py          # Pointer read/write helpers
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
tests/
    pytest suite
docs/
    developer notes and refactor roadmap
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

`wf-orca-thermo-array` runs a per-conformer compound protocol inside a Slurm array task. Each task may
invoke ORCA multiple times sequentially:

- `orca_thermo.inp` for frequency / thermochemistry / high-level SP
- `orca_nmr_h.inp` for proton shieldings
- `orca_nmr_c.inp` for carbon shieldings
- `orca_nmr_j.inp` for J-couplings

The NMR calculations are separate ORCA processes so method-state flags, dispersion settings,
nonlocal correlation flags, and basis choices do not leak between chemically unrelated functionals.

The node emits raw ORCA outputs. `wf-nmr-aggregate` handles parsing, Boltzmann population weighting,
and linear-scaling calibration.

## Standalone output viewers

This repository also builds standalone downloadable viewer bundles for GUI output nodes:

- ensemble viewer
- single-geometry viewer

These viewers are script-backed Layout nodes. The GUI node shim calls into the installed
`scripps_workflow` package, which builds a local ZIP containing `index.html`, static JS/CSS assets,
embedded payload metadata, and molecule/conformer data.

The viewer bundle is intended to be downloaded from the workflow run and opened locally in a browser.

## Cache and database behavior

Cache-aware nodes may ask the `nmr-data` database whether an expensive deterministic result already
exists. This is an optimization.

Required behavior:

- cache hit: emit cached manifest / reuse existing artifacts
- cache miss: compute normally
- database unavailable: warn and compute normally

This prevents temporary PostgreSQL outages from breaking calculations.

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
