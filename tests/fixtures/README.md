# Test fixtures

Hand-picked artifacts used as canonical inputs for tests and for
the local-server harnesses under `tools/`.

## `crest_ensemble_33conf.xyz`

Multi-frame XYZ ensemble of 33 conformers (28 atoms each), CREST
output style — energies on the comment line as bare numbers (no
`E ` prefix). ΔE spans 0 → 3.24 kcal/mol relative to the minimum.
The molecule is `CC/C=C\CC1=C(CCC1=O)C` (a cyclopentenone with a
methyl + propenyl side chain).

Used by:
* `tools/test_ensemble_viewer_locally.py --xyz-file ...` to exercise
  the L2 ensemble_viewer output node against a non-trivial ensemble.
* Future unit tests for any code that parses multi-frame xyz with
  CREST-style comment-line energies.

The first frame's coordinates are the lowest-energy conformer
(E = -36.58944444 hartree). Frame 33 is the highest (E = -36.58427478).

## `sample_experiment/`

Minimal scripps-workflow experiment directory skeleton — only the
single deployment slot the local harness needs to find an output
node:

    sample_experiment/
    └── outputs/
        └── 999/
            └── onlb_sample_block/
                ├── index.html       (placeholder)
                ├── js/viewer.js     (placeholder)
                └── css/styles.css   (placeholder)

The three files are placeholders; the harness's auto-deploy step
overwrites them with the latest generated viewer code on every run.

Use this with the test harnesses to avoid pulling experiment
artifacts from `~/Downloads/`:

    python tools/test_ensemble_viewer_locally.py \
        --experiment-dir tests/fixtures/sample_experiment \
        --xyz-file tests/fixtures/crest_ensemble_33conf.xyz \
        --smiles 'CC/C=C\CC1=C(CCC1=O)C'

The harness's `.harness_staged/` directory is created here on each
run; it's gitignored (see the top-level `.gitignore`).
