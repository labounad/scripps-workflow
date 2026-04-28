# scripps-workflow

A small, HPC-aware framework for chaining quantum-chemistry, cheminformatics, and bioinformatics tools into reproducible pipelines.

## Status

**Pre-alpha.** APIs, wire formats, and node contracts are still moving. Pin a commit if you depend on this.

The engine that schedules these workflows on SLURM is a separate, hosted service (currently at `workflow.scripps.edu`, beta). This repository contains the framework code, node definitions, and workflow definitions — the things the engine round-trips through. You can also run individual nodes by hand: every node is a normal command-line program.

## What it is

Each node is a small Python program with a fixed shape:

- it reads its **config** from `argv` (`key=value` tokens, or JSON for nested config)
- it reads **upstream results** from `stdin` as a one-line JSON pointer to a manifest file
- it writes its **outputs** into a per-call `outputs/` directory
- it emits its own one-line JSON pointer to `stdout`

That's the entire wire protocol. Nodes compose with plain unix pipes:

```
wf-embed smiles="CCO" | wf-xtb theory=gfn2 calculations='["optimize","sp_energy"]'
```

The pointer format is `wf.pointer.v1`:

```json
{"schema": "wf.pointer.v1", "ok": true, "manifest_path": "/abs/path/to/manifest.json"}
```

The manifest (`wf.result.v1`) records the node's inputs, outputs, environment, and per-artifact metadata. A failed node still exits zero and emits a pointer with `ok: false` plus an `error` block in the manifest — the pipeline keeps flowing so downstream nodes can decide how to react. Hard-fail behavior is opt-in via `fail_policy=hard`.

## Repository layout

```
src/scripps_workflow/
    node.py          # Node base class + lifecycle (parse_config, run, manifest emission)
    schema.py        # Manifest / artifact / pointer dataclasses
    pointer.py       # Pointer read/write helpers
    contracts/       # Cross-node contracts (e.g. conformer ensembles)
    nodes/           # Concrete node implementations (smiles_to_3d, xtb_calc, ...)
    tag.py           # Tag-node wiring shims (key=value relay nodes)
nodes/               # Engine-side node descriptors (config schemas, button definitions)
workflows/           # Pre-built workflow definitions
tests/               # Pytest suite
```

## Install (development)

Editable install with the test extras:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
```

Optional extras gate the heavier dependencies so a fresh checkout stays small:

| Extra   | Pulls in                          | Used by                             |
|---------|-----------------------------------|-------------------------------------|
| `test`  | pytest, pyyaml                    | the test suite                      |
| `chem`  | numpy, rdkit                      | cheminformatic / QC nodes           |
| `gui`   | pydantic, pyyaml                  | GUI export/import round-trip tools  |
| `prism` | prism-pruner (requires Python 3.12) | the prism-screen node             |
| `all`   | everything                        | development                         |

The runtime hot path is **stdlib-only by design** — node code itself shouldn't pull in a dependency tree, so missing extras surface as soft import failures recorded in the manifest rather than hard crashes.

## Running tests

```bash
pytest -v
```

Or for a single node:

```bash
pytest tests/test_xtb_calc.py -v
```

The test suite avoids touching real external binaries (xtb, crest, orca) by monkeypatching the single subprocess wrapper in each node.

## Available nodes

| Entry point             | Purpose                                                  |
|-------------------------|----------------------------------------------------------|
| `wf-embed`              | SMILES → 3D coordinates (RDKit)                          |
| `wf-xtb`                | xTB single-point / optimize / gradient / hessian         |
| `wf-crest`              | CREST conformer search                                   |
| `wf-orca-goat`          | ORCA 6.0+ GOAT global conformer search                   |
| `wf-prism`              | RMSD/MoI conformer pruning via prism-pruner              |
| `wf-marc`               | navicat-marc clustering (sibling of `wf-prism`)          |
| `wf-orca-dft-array`     | SLURM-array DFT geometry optimization                    |
| `wf-orca-thermo-array`  | Composite freq + high-level SP (+ optional NMR jobs)     |
| `wf-thermo-aggregate`   | Boltzmann weights + composite Gibbs over an ensemble     |
| `wf-nmr-aggregate`      | Boltzmann-averaged NMR shifts + couplings + calibration  |
| `wf-tag-input`          | Engine wiring shim — relay a single `key=value`          |

The `wf-orca-thermo-array` node runs a per-conformer compound protocol
inside one SLURM array task. Each task may invoke ORCA multiple times
sequentially: `orca_thermo.inp` (freq + high-level SP, internally
chained via `$new_job`) and — when the corresponding `run_shielding_h`
/ `run_shielding_c` / `run_couplings` flag is set — the standalone
inputs `orca_nmr_h.inp`, `orca_nmr_c.inp`, and `orca_nmr_j.inp`. Each
NMR file runs as its own ORCA process so method-state flags
(DFT-NL/VV10, D3/D4, gCP, …) cannot leak between the
chemically-unrelated functionals. The node emits raw ORCA output and
does not apply any NMR referencing or scaling. The downstream
`wf-nmr-aggregate` population-weights the parsed shieldings/J's across
the ensemble and applies a linear-scaling calibration (cheshire for
shifts, Bally/Rablen for couplings) to produce predicted experimental
observables.

## License

MIT. See [`pyproject.toml`](./pyproject.toml).

## Author

Lucas Abounader, The Scripps Research Institute.
