# Refactor Roadmap

This document records the main software-structure issues identified during the
standalone viewer work and proposes an order for addressing them. The goal is to
keep workflow-node behavior stable while making the codebase easier to debug,
test, and extend.

## Completed in the first cleanup pass

### 1. Move viewer assets out of Python string literals

**Problem.** `src/scripps_workflow/output_viewers/assets.py` had grown into a
large Python file containing quoted HTML, CSS, and JavaScript. That made diffs
nearly unreadable and made normal frontend editing awkward.

**Change.** Viewer assets now live under:

```text
src/scripps_workflow/output_viewers/static/
  common/
    styles.css
  ensemble/
    index.html
    js/
      00_bootstrap.js
      10_xyz_parser.js
      20_ensemble_bootstrap.js
      30_sidebar.js
      40_rdkit_2d.js
      50_inset_toggle.js
      60_alignment_math.js
      70_alignment_actions.js
      80_sparkline.js
      90_render_controls.js
      99_measurements_and_3dmol_rendering.js
  geometry/
    index.html
    viewer.js
```

`assets.py` remains as a compatibility shim: it loads these files with
`importlib.resources` and exposes the historical constants consumed by the
bundle builders.

### 2. Split the ensemble viewer JavaScript into source modules

**Problem.** The ensemble viewer is now a real mini frontend application. A
single monolithic `viewer.js` made it hard to work on one feature without
scrolling through unrelated UI, alignment, parsing, and measurement code.

**Change.** The source JavaScript is split into ordered files in
`static/ensemble/js/`. The bundle builder still emits a single `js/viewer.js`
inside the downloadable ZIP, so there is no runtime module-loader requirement
and no change to the user-facing bundle format.

Current source boundaries:

- bootstrap and embedded payload loading
- multi-frame XYZ parsing
- ensemble initialization and energy model
- sidebar/conformer list
- RDKit 2D inset
- inset expansion controls
- alignment math
- alignment actions
- sparkline
- render/view controls
- 3D selection, measurement overlays, and 3Dmol rendering callbacks

### 3. Clean generated/cache files from the project tree

**Problem.** Local artifacts such as `.DS_Store`, `__pycache__`, `.pytest_cache`,
`dist/`, `new_nodes_output/`, and `outputs/` were present in the working tree ZIP.
They are ignored by `.gitignore`, but including them in shared project archives
makes reviews noisy and obscures the real source changes.

**Change.** The project should be distributed from a clean working tree with
ignored cache/generated outputs removed. Generated GUI node ZIPs should be
created locally when needed and committed only by deliberate release policy.


### 4. Unify script-backed Output/Layout node generators

**Problem.** `tools/gen_output_node_ensemble_viewer.py` and
`tools/gen_output_node_geometry_viewer.py` duplicated the same fragile manifest
and `script.py` bootstrap logic.

**Change.** Shared generation now lives in:

```text
tools/gui_export_config.py
tools/output_node_bundle.py
```

The viewer-specific generator scripts are now small declarative wrappers that
provide node names, entrypoints, output filenames, and input specs.

### 5. Centralize HPC/bootstrap configuration

**Problem.** Hardcoded interpreter paths, repo-source fallbacks, author metadata,
and GUI host metadata were embedded directly in each generator.

**Change.** These defaults now live in `tools/gui_export_config.py`. The shared
bootstrap shim still emits a self-contained `script.py`, but changes to the
workflow Python path or fallback repo locations now happen in one file.

### 6. Separate measurement math/state from 3Dmol rendering

**Problem.** The ensemble viewer's measurement feature mixed vector math,
selection state, shape generation, DOM labels, and 3Dmol callbacks in one source
file.

**Change.** The measurement/viewer code is now split into:

```text
91_measurement_math.js      # vector math, distance, angle, dihedral
92_measurement_state.js     # active atoms, labels, radii, selection state
94_measurement_shapes.js    # 3D shape/wedge/bond rendering primitives
96_measurement_ui.js        # measurement label HTML and click-selection actions
99_3dmol_rendering.js       # 3Dmol hover/click callbacks and repaint pass
```

This keeps pure-ish domain math out of the 3Dmol adapter layer and makes the
angle/wedge/dihedral code easier to test and modify.

### 7. Add optional browser-level viewer smoke test

**Problem.** The Python bundle tests verified ZIP creation but did not execute
`index.html` in a browser context.

**Change.** `tests/test_output_viewer_browser_smoke.py` adds an optional
Playwright smoke test marked `browser`. It stubs CDN-loaded 3Dmol/RDKit scripts
so the test checks bundle bootstrap without requiring public internet access.
It is skipped automatically when Playwright or a browser binary is unavailable.

## Completed since the registry refactor

### A. Per-stage producer-side self-registration to `nmr-data`

**Change.** Each compute node now writes its own row + central-tree dir
directly via `nmr_data.registry.register_<stage>(...)` after a successful
run, rather than waiting for the terminal `wf-db-ingest` to bulk-ingest
everything. `wf-db-ingest` survives as a fail-open backstop. See
`docs/registry-design.md` and `docs/registry-verification.md` for
rationale + verification protocol. `tools/inspect_registry_progress.py`
prints the DB + tree state for a given SMILES.

### B. Element-driven basis selection (heavy atoms + ZORA)

**Change.** `wf-orca-thermo-array` scans the upstream geometry for
element symbols and routes each NMR job through
`scripps_workflow.basis_coverage.compute_coverage_decision`:

* HALA-relevant elements (4d/5d TMs, lanthanides, actinides) trigger
  a global swap to `relativistic_basis` (default `def2-ZORA-TZVPP`)
  + `! ZORA` prefix on the job's `!` line. The operator's configured
  basis is discarded for that job.
* Light-heavy elements (Br, I, Se, Sn, ...) outside the configured
  basis trigger per-atom `%basis newgto` supplementation using
  `heavy_atom_basis` (default `def2-TZVPP`). The calibrated light-
  atom basis stays put.

No `system_class` profile — the decision is purely per-element. Two
config knobs (`heavy_atom_basis`, `relativistic_basis`) tune the
supplement / swap targets. The cache identity recorded for
`PredictedRun` distinguishes the three regimes via the basis string
(`6-31G(d,p)` / `6-31G(d,p)+def2-TZVPP/heavy` /
`def2-ZORA-TZVPP`), so the existing basis-keyed `PredictedRunKey`
naturally invalidates without DB changes.

### C. Heteronuclear J auto-detection (¹⁹F / ³¹P)

**Change.** When ¹⁹F or ³¹P is present in the upstream geometry,
`wf-orca-thermo-array` auto-appends `"all F"` / `"all P"` to
`coupling_pairs` so ORCA computes the cross-element J's;
`wf-nmr-aggregate` mirrors the detection on its own upstream and
auto-extends `mnova_heteronuclear_partners` so the simulated H/C
spectra render with realistic splitting. New `auto_heteronuclear`
knob (default true) on both nodes; set false for the {¹⁹F}/{³¹P}
decoupled-spectrum view.

### D. Equivalence-group annotations on aggregator CSVs

**Change.** `predicted_shifts.csv` and `predicted_couplings.csv` carry
per-atom rows (unchanged) plus per-group annotation columns:
`group_label`, `group_tier`, `group_n_atoms`, `group_atom_indices`,
and group-averaged σ/δ values. Collapse rows by `group_label` for a
spectrum-comparison view. Singleton fallback when no SMILES is
available so the CSV shape is invariant.

### E. ¹³C lab-fit calibration default

**Change.** Default ¹³C calibration for `wB97X-D/6-31G(d,p)/CHCl3`
swapped from cheshire (-1.0501, 187.25) to a lab-fit recalibration
(-0.9781, 197.67) against organopalladium experimental data
(R²=0.9986, RMSE=1.51 ppm). The original cheshire row is still
reachable by overriding solvent to e.g. `CHCl3_CHESHIRE_2006`. ¹H
calibration retains the cheshire WP04 values.

### F. SSH-driven HPC integration test scaffold

**Change.** `tests/test_hpc_integration.py` adds a Playwright-style
opt-in test layer that reaches over SSH to a real cluster. Gated by
`@pytest.mark.hpc` plus a `SCRIPPS_WORKFLOW_HPC_HOST` env var; default
`pytest` runs skip silently. Smoke layer covers SSH, env activation,
DB connectivity, alembic schema head, and a `wf-embed` round-trip.

## Remaining refactor candidates

### 8. Remove stale GUI-viewer/inport-era artifacts

The current viewer strategy is a script-backed standalone ZIP bundle. Any older
fixture output or experimental node bundle containing `opaat`,
`workflow_backend`, `get_inputs_for_output_node`, staged iframe data, or polling
logic should either be deleted or moved into a clearly named deprecated fixture
folder.

Recommended policy:

```text
tests/fixtures/deprecated_gui_inport_viewer/
```

for anything kept only as historical/debug context.

### 9. Refactor large node modules along internal boundaries

Some node modules are very large and should eventually be split by responsibility.
Examples include NMR aggregation, ORCA array nodes, ORCA helpers, and CREST.
Suggested boundaries:

```text
config parsing
input artifact resolution
job/input-file generation
execution/submission
monitoring
output parsing
manifest writing
CLI main()
```

Do this incrementally, one node at a time, with tests passing after each split.

### 10. Consolidate shared conformer and XYZ utilities

Several modules know how to split multi-frame XYZ files, select conformers,
resolve artifacts, and infer labels. The registry refactor doubled-down on
this duplication: `_walk_upstream_for_step`, `_walk_upstream_for_smiles`,
`_upstream_xyz_paths`, and element-detection scanners now exist in roughly
five node modules (crest, orca_goat, orca_dft_array, orca_thermo_array,
nmr_aggregate). Move these shared utilities toward:

```text
src/scripps_workflow/conformers/
  xyz.py
  ensemble.py
  artifacts.py
src/scripps_workflow/upstream_walk.py    # manifest-chain traversal
src/scripps_workflow/element_detect.py   # xyz element scanning
```

Then `artifact_resolver.py`, `extract_conformers.py`, and conformer-producing
nodes can share one implementation.

### 11. Make artifact records typed

Manifest handling is still stringly typed:

```python
item["path_abs"]
item["label"]
item["format"]
bucket = "xyz_ensemble"
```

Introduce dataclasses such as:

```python
ArtifactRecord
ArtifactBucket
ResolvedArtifact
ResolvedXyz
```

This should reduce uncertainty about bucket shapes and make refactors safer.

### 12. Isolate legacy compatibility

Compatibility logic is useful, but it should live at boundaries rather than
inside core logic. Consider modules such as:

```text
schema_compat.py
legacy_manifest_upgrade.py
orca_legacy_parsing.py
```

Core code can then operate on normalized modern records.

**Partial progress:** `nmr_data.ingest.ingest_nmr_aggregate_result` now
delegates to `nmr_data.registry.register_*` and survives as a thin
compatibility shim, but it still lives in `ingest.py` rather than a
dedicated `legacy_*.py` module.

### 13. Remove the hardcoded JS atom-map fallback

The viewer still has an old hardcoded fixture atom map for a specific SMILES.
Now that Python-side RDKit atom mapping exists, that should eventually be
removed or confined to a test-only fixture.

### 14. Clarify generated-vs-source artifact policy

Decide which generated artifacts belong in Git. Recommended default:

```text
src/                 source
src/.../static/      source frontend assets
tools/               source generators
tests/fixtures/      small committed fixtures only
dist/                generated release artifacts; normally ignored
new_nodes_output/    generated local node bundles; normally ignored
outputs/             run artifacts; ignored
```

If generated node ZIPs are committed for a release, document that as a deliberate
release step.

### 15. Add a developer lifecycle document

Document the current development loop:

```bash
pytest
python tools/gen_output_node_ensemble_viewer.py
python tools/gen_output_node_geometry_viewer.py
git add ...
git commit -m "..."
git push origin main

# on HPC
cd /gpfs/group/shenvi/code/scripps-workflow
git pull
pip install -e .
```

Also clarify when GUI node re-import is required:

- **Not required:** implementation-only repo changes used by script-backed nodes.
- **Required:** GUI-facing changes such as input names, node category, output file
  declarations, or shim entrypoint changes.

## Near-term recommended next steps

1. Pull the duplicated upstream-walk + element-detect helpers into
   `scripps_workflow.upstream_walk` + `scripps_workflow.element_detect`
   (item 10). Five node modules share copies right now.
2. Move the legacy `ingest_nmr_aggregate_result` compatibility shim into a
   dedicated module so `nmr_data.ingest` only contains current-shape
   helpers (item 12).
3. Add DB-first calibration lookup so `nmr_aggregate` reads slope/intercept
   from the `calibrations` table instead of the hardcoded `NMR_CALIBRATION`
   dict. Removes the source-edit-redeploy loop for new calibrations.
4. Build `wf-fit-calibration` CLI to fit slope/intercept from
   (σ_calc, δ_exp) pairs and write a `Calibration` row; closes the loop
   between predicted runs and experimental observations.
5. Translate `docs/registry-verification.md` into pytest as an
   "executable runbook" (the 4-claim end-to-end protocol). Currently
   manual; should be automated.
6. Decide whether to commit any generated node ZIPs or keep them entirely local
   (item 14).
7. Remove or quarantine deprecated GUI-inport-era fixtures (item 8).
