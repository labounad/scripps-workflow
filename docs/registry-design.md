# Per-stage Self-Registration to nmr-data

Design note for the decentralization of `wf-db-ingest`.

## Goal

Each cacheable compute stage of the NMR Predictor pipeline writes its own
identity row and copies its artifacts into the central data tree as the stage
finishes, instead of waiting for the terminal `wf-db-ingest` node to ingest
the whole pipeline at once.

The motivation is operational, not schematic:

- A pipeline that dies at NMR aggregate currently loses *all* upstream DB
  state. A re-run can't cache-hit the DFT optimization that already ran.
- Central-tree population happens only at the end. Long-running DFT and
  thermo arrays leave no breadcrumb under `NMR_HPC_DATA_ROOT/` until aggregate
  finishes.
- The shape of `wf-db-ingest` (one ~445-line function taking six manifest
  dicts via `**kwargs`) couples the data layer to the workflow layer too
  tightly. Each stage already has a clean cache-key story; the writer side
  should mirror it.

## Current shape

`nmr_data.ingest` already has well-typed, idempotent per-stage primitives:

| Stage           | Row helper                          | Copy helper                |
|-----------------|-------------------------------------|----------------------------|
| ensemble        | `get_or_create_conformer_ensemble`  | `copy_ensemble_artifacts`  |
| dft_run         | `get_or_create_dft_run`             | `copy_dft_run_artifacts`   |
| thermo_run      | `get_or_create_thermo_run`          | `copy_thermo_run_artifacts`|
| predicted_run   | `get_or_create_predicted_run`       | `copy_predicted_run_artifacts` |

These primitives are tested, idempotent on fingerprint, and already aware of
the per-stage central-tree layout. The only reason they all fire from one
place today is that `ingest_nmr_aggregate_result` bundles them. The hard
work is done.

The reader side of the cache (`nmr_data.cache.find_*`, used by
`orca_dft_array`, `orca_goat`, `orca_thermo_array`, `crest`) already
fingerprint-resolves parents. There is no writer-side equivalent.

## Target shape

### `nmr_data.registry`

A new module exposing four entry points, one per stage. Each takes the
producing node's manifest material plus its **parent's fingerprint** (which
the node already has from its upstream manifest pointer), and wraps the
matching `get_or_create_*` + `copy_*_artifacts` pair under one session.

```python
def register_ensemble(
    *,
    smiles: str,
    ensemble_key: EnsembleKey,
    outputs_dir: Path,
    hpc_data_root: str | None = None,
    project_id: Any | None = None,
    name: str | None = None,
) -> RegistryResult: ...

def register_dft_run(
    *,
    parent_ensemble_fingerprint: str,
    dft_key: DftRunKey,
    outputs_dir: Path,
    conformer_records: list[dict],
    hpc_data_root: str | None = None,
) -> RegistryResult: ...

def register_thermo_run(
    *,
    parent_dft_run_fingerprint: str,
    thermo_key: ThermoKey,
    outputs_dir: Path,
    conformer_thermo_rows: list[dict],
    hpc_data_root: str | None = None,
) -> RegistryResult: ...

def register_predicted_run(
    *,
    smiles: str,
    nmr_run_params: dict,
    shifts_csv_path: Path,
    couplings_csv_path: Path,
    outputs_dir: Path,
    conformer_records: list[dict],
    hpc_data_root: str | None = None,
    project_id: Any | None = None,
) -> RegistryResult: ...
```

Common result shape:

```python
@dataclass
class RegistryResult:
    ok: bool
    status: str                   # "created" | "reused" | "skipped" | "failed"
    row_id: Any | None            # UUID of the registered row
    fingerprint: str | None       # provenance_fingerprint of the registered row
    central_tree_path: str | None # relative path under NMR_HPC_DATA_ROOT
    notes: list[str]
```

Idempotency is preserved by the underlying `get_or_create_*` calls; re-runs
return the same row and `status="reused"`. Parent-fingerprint mismatch (e.g.
DFT registration without a registered ensemble) returns
`ok=False, status="skipped"` with a note — never raises — so the node hook
can log + continue without aborting the run.

### `Node._try_register_to_nmr_data(...)`

Producing nodes call a single base-class helper. The pattern mirrors the
cache fail-open path in `caf23ff`:

```python
def _try_register_to_nmr_data(
    self,
    stage: str,                       # "ensemble" | "dft_run" | "thermo_run" | "predicted_run"
    *,
    ctx: NodeContext,
    **kwargs,
) -> dict[str, Any] | None:
    """Best-effort registration. Failure is non-fatal and logged."""
    try:
        from nmr_data import registry
        fn = getattr(registry, f"register_{stage}")
    except Exception as e:
        logging_utils.log_warn(
            f"registry({stage}): import unavailable, skipping registration: "
            f"{type(e).__name__}: {e}"
        )
        return None
    try:
        result = fn(hpc_data_root=os.environ.get("NMR_HPC_DATA_ROOT"), **kwargs)
        return _registry_result_to_dict(result)
    except Exception as e:
        logging_utils.log_warn(
            f"registry({stage}): registration raised; treating as no-op: "
            f"{type(e).__name__}: {e}"
        )
        return {"ok": False, "status": "failed", "error": str(e)}
```

The returned dict (or `None`) is attached to the node's emitted manifest
under a `"registry"` block so downstream observers (and a future audit
tool) can see what landed in the DB.

### `wf-db-ingest` after the refactor

Once `wf-nmr-aggregate` self-registers the predicted_run + its scalars,
`wf-db-ingest` no longer needs to ingest anything. It shrinks to:

1. Verify the predicted_run row exists for the molecule + method tuple.
2. Optionally attach late-binding metadata (project_id, hpc_job_id) the
   producing nodes could not know.
3. Emit a summary pointer.

The node stays in the catalog so existing GUI pipelines don't break, but
its `run()` body collapses to ~20 lines.

## Failure semantics

Fail-open across the board, matching the existing cache fail-open policy:

- DB connection error: log warning, continue, the node's compute output is
  unchanged. The node's manifest records `registry.ok = False`.
- Schema error: same.
- Missing parent fingerprint: `RegistryResult(ok=False, status="skipped")`,
  no exception. (This is a real possibility during rollout when an upstream
  node was run before its self-register hook landed.)
- Idempotent reuse of an existing row: `status="reused"`, treated as success.

This is the right policy for production but means a silent registry bug
could hide for a while. The end-to-end verification step (task #84) is the
checkpoint that catches that class of bug before rollout.

## Phasing

All four stages in one design pass (per #73 decision):

1. `nmr_data.registry` module + tests (#74, #75)
2. `ingest_nmr_aggregate_result` re-pointed at register_* internally (#76);
   external signature unchanged, all existing tests pass.
3. `Node._try_register_to_nmr_data` base-class helper (#77).
4. Wire all four producing stages in one PR (#78–#82): crest, orca_goat,
   orca_dft_array, orca_thermo_array, nmr_aggregate.
5. Shrink `wf-db-ingest` (#83). Keep the node, kill the body.
6. End-to-end verification on `aggregate_demo` (#84).

Each step is independently reversible. After (2), `db_ingest` keeps working
unchanged because the primitives it calls now route through `registry`
internally. After (4) plus (5), the pipeline is fully decentralized but the
old `db_ingest` node is still a no-op safety net.

## Open questions for future work

- **Pruning provenance.** PRISM/MARC don't get their own DB row in this
  design — the v6.6-v2 schema only attaches conformers to `DftRun`, so the
  pruning step is transparent. If a future use case wants to record "which
  CREST conformers got dropped by PRISM," that's an additive ensemble-level
  table, not a structural change here.
- **In-progress markers.** This design only writes rows on stage success.
  A future enhancement could insert a "stage_attempt" row at start and
  mark it completed/failed at the end, giving operators visibility into
  jobs that crashed before finishing. Out of scope for the first pass.
- **Registry-driven cache invalidation.** Today fingerprints are the only
  invalidation tool. Once registry is in, a `nmr_data.registry.invalidate`
  CLI for hand-pulling stale rows becomes natural to add.
