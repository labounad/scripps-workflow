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
resolve artifacts, and infer labels. Move these shared utilities toward:

```text
src/scripps_workflow/conformers/
  xyz.py
  ensemble.py
  artifacts.py
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

1. Verify that the static-asset refactor produces identical viewer bundles.
2. Decide whether to commit any generated node ZIPs or keep them entirely local.
3. Remove or quarantine deprecated GUI-inport-era fixtures.
4. Start splitting large chemistry node modules along their internal boundaries.
