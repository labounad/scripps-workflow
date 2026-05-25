# Registry Verification — End-to-End Checklist

The producer-side self-registration shipped in tasks #73–#83 is now in
place across all five producing nodes. This document is the verification
runbook for one full pass against the HPC, exercising the four claims
the design rests on:

1. **Per-stage rows land progressively.** A row appears in the DB as
   soon as its producing node finishes, not at the end via
   `wf-db-ingest`.
2. **Central-tree dirs populate per stage.** Each producing node copies
   its artifacts into `NMR_HPC_DATA_ROOT/<inchikey>/<stage>/<uuid>/`
   right after it succeeds.
3. **Mid-pipeline kill leaves a consistent partial state.** A pipeline
   killed at, say, the NMR aggregate step has every upstream stage
   already in the DB and the central tree, so a re-run cache-hits.
4. **Full re-run is a no-op.** Every `register_*` returns `"reused"`
   and `n_files_copied=0` on a re-run with the same fingerprints.

## Preflight

Activate the workflow runtime env on a login node:

```bash
micromamba activate /gpfs/group/shenvi/envs/workflow312
echo "$NMR_DATABASE_URL"
echo "$NMR_HPC_DATA_ROOT"
```

Both env vars should be set. If `NMR_DATABASE_URL` is unset every
node will log `registry(<stage>): NMR_DATABASE_URL unset; skipping
registration` and the verification is meaningless.

Confirm the database is reachable:

```bash
psql "$NMR_DATABASE_URL" -c "SELECT current_database(), current_user;"
```

Confirm the schema is at head:

```bash
cd /gpfs/group/shenvi/code/nmr-data
alembic current
alembic upgrade head    # no-op if already at head
```

Choose a verification SMILES that does NOT already exist in the DB.
Methanol is convenient — small, fast, RDKit-stable:

```text
CO    inchikey  OKKJLVBELUTLKV-UHFFFAOYSA-N
```

Confirm it's absent:

```bash
psql "$NMR_DATABASE_URL" -c \
  "SELECT id, inchikey FROM molecules WHERE inchikey = 'OKKJLVBELUTLKV-UHFFFAOYSA-N';"
```

Expect 0 rows. If non-zero, either pick a different SMILES, or
explicitly delete the existing graph:

```sql
DELETE FROM molecules WHERE inchikey = 'OKKJLVBELUTLKV-UHFFFAOYSA-N';
-- (cascades via the FK chain set up in commit 5f3c398)
```

## Run 1 — Cold pipeline (claim 1 + 2)

Kick off the full pipeline against the chosen SMILES. The exact GUI
recipe is `NMR Predictor` (built by `workflows/nmr_predictor.py`).

After each producing node lands a manifest, check that its own row +
central-tree dir appear before the next node finishes. The shape of
each check:

### CREST

After CREST's per-call dir contains `outputs/manifest.json`:

```bash
psql "$NMR_DATABASE_URL" -c \
  "SELECT id, provenance_fingerprint, ensemble_path, n_conformers
   FROM conformer_ensembles
   WHERE molecule_id = (
     SELECT id FROM molecules WHERE inchikey = 'OKKJLVBELUTLKV-UHFFFAOYSA-N'
   );"
```

Expect exactly 1 row, with `ensemble_path` non-NULL.

```bash
ls "$NMR_HPC_DATA_ROOT/OKKJLVBELUTLKV-UHFFFAOYSA-N/ensembles/"
ls "$NMR_HPC_DATA_ROOT/OKKJLVBELUTLKV-UHFFFAOYSA-N/ensembles/<uuid>/conformers/"
```

Expect the ensemble's UUID dir and one `conf_NNNN/conf_NNNN.xyz` per
CREST conformer, plus a `conformer_energies.json` sidecar.

CREST's own manifest should now carry a `registry` block:

```bash
python -c "
import json
m = json.load(open('<crest_call_dir>/outputs/manifest.json'))
print(json.dumps(m['inputs'].get('registry', '<missing>'), indent=2))
"
```

Expect: `{"ensemble": {"ok": true, "status": "created", "row_id":
"<uuid>", "fingerprint": "<hex>", "central_tree_path": "...", ...}}`.

### orca_dft_array

After the DFT array completes, query for the dft_run row:

```bash
psql "$NMR_DATABASE_URL" -c \
  "SELECT id, provenance_fingerprint, dft_run_path, dft_opt_method
   FROM dft_runs
   ORDER BY created_at DESC LIMIT 1;"

ls "$NMR_HPC_DATA_ROOT/OKKJLVBELUTLKV-UHFFFAOYSA-N/dft_runs/"
```

Expect 1 row with `dft_opt_method='r2scan-3c'` and a matching
central-tree dir holding DFT-optimized geometries + `orca_opt.out.gz`.

### orca_thermo_array

After the thermo array completes:

```bash
psql "$NMR_DATABASE_URL" -c \
  "SELECT id, provenance_fingerprint, thermo_run_path, thermo_method
   FROM thermo_runs
   ORDER BY created_at DESC LIMIT 1;"

ls "$NMR_HPC_DATA_ROOT/OKKJLVBELUTLKV-UHFFFAOYSA-N/thermo_runs/"
```

Expect 1 row plus a `thermo_runs/<uuid>/conformers/conf_*/orca_thermo.out.gz`
tree.

Known limitation: the thermo_aggregate manifest + summary CSV land in
the central tree only after `wf-db-ingest` runs (the dir already
exists by then, blocking the incremental copy). This is captured as an
open follow-up in `registry-design.md` — fix is to make
`copy_thermo_run_artifacts` incremental.

### nmr_aggregate

After the NMR aggregate finishes:

```bash
psql "$NMR_DATABASE_URL" -c \
  "SELECT id, run_root_path, n_conformers_used,
          (SELECT COUNT(*) FROM predicted_shifts WHERE run_id = pr.id) AS n_shifts,
          (SELECT COUNT(*) FROM predicted_couplings WHERE run_id = pr.id) AS n_couplings
   FROM predicted_runs pr
   ORDER BY created_at DESC LIMIT 1;"

ls "$NMR_HPC_DATA_ROOT/OKKJLVBELUTLKV-UHFFFAOYSA-N/predicted_runs/"
```

Expect 1 row with `n_shifts > 0` and `n_couplings > 0`, plus a
`predicted_runs/<uuid>/` dir with `predicted_shifts.csv`,
`predicted_couplings.csv`, mnova XMLs, and `predicted_structure_*.svg/html`.

### wf-db-ingest

After the backstop runs, the stderr log should say:

```text
[7/8] DB write OK: molecule=OKKJLVBELUTLKV-... [existing], run=... [existing]
      backstop no-op — predicted_run was already registered by wf-nmr-aggregate
      ensemble dir:      .../ensembles/<uuid> (0 files)
      dft_run dir:       .../dft_runs/<uuid> (0 files)
      thermo_run dir:    .../thermo_runs/<uuid> (0 files)
      predicted_run dir: .../predicted_runs/<uuid> (0 files)
```

`(0 files)` across all four lines is the success signal for claim 1
and claim 4: the producing nodes already wrote everything, and the
backstop is a true no-op.

If you see `[NEW]` on any of those lines, it means a producing node's
self-registration was skipped — check the producing node's stderr for
a `registry(<stage>):` `[INFO]` / `[WARN]` line explaining why.

## Run 2 — Re-run from clean session (claim 4)

Re-fire the same pipeline against the same SMILES. Expected behavior:

- Every producing node's `_maybe_emit_cached_manifest_*` short-circuits
  (cache hits all the way down).
- The pipeline finishes in seconds, not hours.
- No new rows in any table.
- `wf-db-ingest` logs the same `(0 files)` no-op summary as Run 1.

If any compute node fires fresh, the cache key derivation has drifted
— check the upstream-walked inputs against the writer-side
`build_*_ensemble_key` / `build_dft_run_key` / `build_thermo_run_key`
to find the field that changed.

## Run 3 — Kill mid-pipeline + restart (claim 3)

Pick a SMILES that doesn't yet exist in the DB. Kick off the pipeline.
**Cancel the SLURM job** after `orca_dft_array` finishes but before
`orca_thermo_array` does (the engine will detect the cancel and stop
the pipeline at the next node boundary).

Confirm the partial DB + central-tree state:

```bash
psql "$NMR_DATABASE_URL" -c \
  "SELECT 'ensemble' AS stage, COUNT(*) FROM conformer_ensembles WHERE molecule_id = (SELECT id FROM molecules WHERE inchikey = '<NEW_INCHIKEY>')
   UNION ALL
   SELECT 'dft_run',     COUNT(*) FROM dft_runs       WHERE ensemble_id IN (SELECT id FROM conformer_ensembles WHERE molecule_id = (SELECT id FROM molecules WHERE inchikey = '<NEW_INCHIKEY>'))
   UNION ALL
   SELECT 'thermo_run',  COUNT(*) FROM thermo_runs    WHERE dft_run_id   IN (SELECT id FROM dft_runs WHERE ensemble_id IN (SELECT id FROM conformer_ensembles WHERE molecule_id = (SELECT id FROM molecules WHERE inchikey = '<NEW_INCHIKEY>')))
   UNION ALL
   SELECT 'predicted',   COUNT(*) FROM predicted_runs WHERE molecule_id = (SELECT id FROM molecules WHERE inchikey = '<NEW_INCHIKEY>');"
```

Expect:

```text
 stage     | count
-----------|-------
 ensemble  | 1
 dft_run   | 1
 thermo_run| 0
 predicted | 0
```

Then re-fire the same pipeline. CREST and orca_dft_array both
cache-hit (their rows exist); orca_thermo_array and downstream compute
normally. End state: full row graph + central tree.

## Failure modes worth catching

| Symptom in db_ingest log                              | Likely cause                                                                 |
|-------------------------------------------------------|------------------------------------------------------------------------------|
| `ensemble: NEW` on Run 1's backstop                   | CREST's `_maybe_register_ensemble` is being skipped; check its stderr        |
| `(N files)` non-zero on a stage during Run 1 backstop | Producing node didn't copy to central tree; check stage's stderr             |
| `dft_run: NEW` on Run 2 (re-run)                      | Cache miss in `_maybe_emit_cached_manifest_dft` — key drift                  |
| Per-conformer thermo rows missing post-Run 1          | `parent_thermo_run_fingerprint` resolution failed in `nmr_aggregate`         |
| `predicted_run: NEW` on Run 2 but other stages reused | nmr_aggregate's writer / db_ingest's writer compute different method tuples  |

## Inspecting state directly

`tools/inspect_registry_progress.py <smiles>` prints the current DB +
central-tree state for a given SMILES — useful for quick health checks
without writing SQL by hand.
