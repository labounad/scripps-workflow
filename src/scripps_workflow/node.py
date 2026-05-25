"""Node base class — boilerplate every node would otherwise repeat.

A workflow node script's job is simple in the abstract:

    1. Parse argv[1] as an upstream pointer (or accept being a source node
       with no upstream).
    2. Parse argv[2:] as a config dict.
    3. Do work, producing files in ``outputs/`` under the call directory.
    4. Write ``outputs/manifest.json`` describing what happened, even if
       the run failed.
    5. Print exactly one line of pointer JSON to stdout.
    6. Exit 0 (soft-fail) — unless explicitly configured otherwise.

This module turns that protocol into a thin context-manager + base class.
A concrete node extends :class:`Node`, implements :meth:`Node.run`, and
that's it; the framework guarantees:

    * The manifest is written (even if ``run`` raises before returning).
    * The pointer is printed (even if ``run`` raises).
    * ``inputs.raw_argv`` is captured verbatim.
    * ``environment`` (python / python_exe / platform / host) is filled.
    * ``runtime_seconds`` is measured.
    * ``ok`` reflects whether ``run`` raised, plus whatever the node sets
      explicitly via ``ctx.fail()``.

This is the **one place** the load-bearing pointer / manifest invariants
live. Touching them anywhere else is a smell.
"""

from __future__ import annotations

import platform as _platform
import socket
import sys
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .logging_utils import log_error, log_info, log_warn
from .parsing import parse_kv_or_json
from .pointer import Pointer, PointerError, load_pointer
from .schema import (
    ArtifactRecord,
    EnvironmentInfo,
    Manifest,
    UpstreamRef,
)


@dataclass
class NodeContext:
    """Per-call mutable state passed to :meth:`Node.run`.

    Holds the partially-built manifest, the parsed config, the upstream
    pointer (if any), and convenience paths. The node's ``run`` mutates
    ``manifest`` in place (adding artifacts, parsed inputs) and does not
    have to worry about finalization — :class:`Node` handles that.
    """

    cwd: Path
    outputs_dir: Path
    manifest_path: Path
    raw_argv: list[str]
    config: dict[str, Any]
    upstream_pointer: Pointer | None
    upstream_manifest: Manifest | None
    manifest: Manifest
    started_at_unix: int
    started_at_perf: float
    fail_policy: str = "soft"

    # ---------- convenience setters ----------

    def set_input(self, key: str, value: Any) -> None:
        """Record a parsed input value into ``manifest.inputs``.

        Use this for everything the node interpreted from ``argv`` — it is
        what makes manifests self-describing without reverse-engineering
        ``raw_argv``.
        """
        self.manifest.inputs[key] = value

    def set_inputs(self, **kwargs: Any) -> None:
        """Bulk variant of :meth:`set_input`."""
        for k, v in kwargs.items():
            self.set_input(k, v)

    def add_artifact(
        self,
        bucket: str,
        record: ArtifactRecord | dict[str, Any],
    ) -> None:
        self.manifest.add_artifact(bucket, record)

    def fail(self, error: str, **extra: Any) -> None:
        """Record a structured failure AND flip ``ok=False``.

        The two-step ``add_failure`` + ``ok = False`` happens together here
        because forgetting one of the two is the most common manifest bug.
        """
        self.manifest.add_failure(error, **extra)
        self.manifest.ok = False
        log_error(error)


class Node(ABC):
    """Base class for workflow nodes.

    Subclasses must define:
        step:        unique step name written into the manifest (snake_case)
        accepts_upstream:
                     ``True`` for chain nodes (xtb, crest, prism, ...);
                     ``False`` for source nodes (smiles_to_3d, tag_input).

    Subclasses implement:
        :meth:`run` — the actual work. Receives a fully-prepared
        :class:`NodeContext`.

    Subclasses may override:
        :meth:`parse_config` — to coerce raw config to typed values.
            Default just returns the parsed dict.

    The class method :meth:`Node.invoke` is the engine entrypoint:
    framework users wire ``main = Node.invoke_factory(MyNode)`` and the
    ``wf-*`` console script does the rest.
    """

    #: Unique step name (snake_case). Written into ``manifest.step``.
    step: str = ""

    #: ``True`` if the node consumes ``argv[1]`` as a pointer JSON.
    accepts_upstream: bool = True

    #: ``True`` if ``argv[1]`` (the pointer) is required even though the
    #: engine declares all inputs ``required=0``. Set ``False`` only on
    #: source nodes (smiles_to_3d, tag_input).
    requires_upstream: bool = True

    # ---------- subclass hooks ----------

    @abstractmethod
    def run(self, ctx: NodeContext) -> None:
        """Perform the node's work. Mutate ``ctx.manifest`` in place.

        Raising an exception is fine: the framework catches it, records a
        failure, and writes the manifest before exiting. Soft-fail
        semantics are preserved.
        """
        raise NotImplementedError

    def parse_config(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Hook for typed config coercion. Default = identity.

        Override in subclasses to return a typed dict (e.g. ints, floats,
        bools coerced) and call ``ctx.set_inputs(**typed)`` from ``run``.
        """
        return dict(raw)

    # ---------- producer-side self-registration to nmr-data ----------

    #: Stages that producing nodes may self-register. Mirrors the four
    #: cache-key tiers in :mod:`nmr_data.registry`. Keep in sync with the
    #: registry module's public entry points.
    _REGISTRY_STAGES: frozenset[str] = frozenset(
        {"ensemble", "dft_run", "thermo_run", "predicted_run"}
    )

    def _try_register_to_nmr_data(
        self,
        stage: str,
        *,
        ctx: NodeContext,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Best-effort per-stage self-registration to nmr-data.

        Calls ``nmr_data.registry.register_<stage>(**kwargs)`` after the
        producing node's compute step succeeds. The result is normalized
        to a dict and stashed under ``manifest.inputs["registry"][stage]``
        so the operator (and downstream tooling) can see what landed in
        the DB. The same dict is also returned for callers that want to
        inspect it.

        This method NEVER raises. Failure modes (env unset, nmr_data
        unimportable, DB down, missing parent fingerprint) all map to a
        result dict with ``ok=False`` and a ``reason`` field, plus a
        warning on stderr. Matches the cache fail-open policy in
        ``caf23ff``: a degraded data layer never breaks a running
        pipeline.

        :param stage: One of :data:`_REGISTRY_STAGES`. Determines which
            ``register_*`` entry point to call.
        :param ctx: Node context. Used only to stash the result into
            the manifest.
        :param kwargs: Forwarded verbatim to
            ``nmr_data.registry.register_<stage>``. See that module for
            per-stage required and optional kwargs. ``hpc_data_root`` is
            auto-filled from ``NMR_HPC_DATA_ROOT`` if not provided.

        :returns: Normalized result dict with keys
            ``{stage, ok, status, row_id, fingerprint, central_tree_path,
            n_files_copied, notes, extra, reason?}``. Always present;
            never ``None``.
        """
        import os

        if stage not in self._REGISTRY_STAGES:
            raise ValueError(
                f"Unknown registration stage {stage!r}; "
                f"expected one of {sorted(self._REGISTRY_STAGES)}"
            )

        def _stash(result: dict[str, Any]) -> dict[str, Any]:
            """Tuck the result under ``manifest.inputs.registry[stage]``."""
            reg_bucket = ctx.manifest.inputs.setdefault("registry", {})
            if not isinstance(reg_bucket, dict):
                # Existing non-dict value at "registry" — leave it alone
                # and return without stashing. Should not happen in
                # normal use; defensive against external manipulation.
                return result
            reg_bucket[stage] = result
            return result

        database_url = os.environ.get("NMR_DATABASE_URL")
        if not database_url:
            log_info(
                f"registry({stage}): NMR_DATABASE_URL unset; "
                f"skipping registration"
            )
            return _stash({
                "stage": stage,
                "ok": False,
                "status": "skipped",
                "reason": "NMR_DATABASE_URL unset",
            })

        try:
            from nmr_data import registry  # type: ignore[import-not-found]
        except Exception as e:
            log_warn(
                f"registry({stage}): nmr_data import failed, "
                f"skipping registration: {type(e).__name__}: {e}"
            )
            return _stash({
                "stage": stage,
                "ok": False,
                "status": "failed",
                "reason": f"import: {type(e).__name__}: {e}",
            })

        try:
            fn = getattr(registry, f"register_{stage}")
            result = fn(**kwargs)
        except Exception as e:
            log_warn(
                f"registry({stage}): registration raised, "
                f"treating as no-op: {type(e).__name__}: {e}"
            )
            return _stash({
                "stage": stage,
                "ok": False,
                "status": "failed",
                "reason": f"{type(e).__name__}: {e}",
            })

        # Normalize the RegistryResult dataclass → JSON-able dict for
        # manifest stash. UUIDs become strings; everything else is
        # already JSON-friendly per the registry module's contract.
        normalized = {
            "stage": stage,
            "ok": bool(result.ok),
            "status": str(result.status),
            "row_id": str(result.row_id) if result.row_id is not None else None,
            "fingerprint": result.fingerprint,
            "central_tree_path": result.central_tree_path,
            "n_files_copied": int(result.n_files_copied),
            "notes": list(result.notes),
            "extra": dict(result.extra),
        }
        if result.ok:
            log_info(
                f"registry({stage}): {result.status} "
                f"row={normalized['row_id']} "
                f"path={result.central_tree_path or '-'} "
                f"files={result.n_files_copied}"
            )
        else:
            log_warn(
                f"registry({stage}): {result.status} — {'; '.join(result.notes) or '(no notes)'}"
            )
        return _stash(normalized)

    # ---------- engine entry ----------

    def invoke(self, argv: Sequence[str] | None = None) -> int:
        """Run the node end-to-end. Returns process exit code.

        Always returns 0 by default (soft-fail) — set
        ``self.fail_policy = "hard"`` (or pass ``fail_policy=hard`` in
        config) to exit 1 on failure.
        """
        argv_list = list(sys.argv if argv is None else argv)

        # Always work in the engine-provided call directory. Each call gets
        # its own folder under ``calls/`` on the engine side, so this is
        # safe to assume.
        cwd = Path.cwd().resolve()
        outputs_dir = cwd / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = outputs_dir / "manifest.json"

        started_at_perf = time.perf_counter()
        started_at_unix = int(time.time())

        upstream_pointer: Pointer | None = None
        upstream_manifest: Manifest | None = None
        upstream_ref = UpstreamRef()

        manifest = Manifest.skeleton(
            step=self.step or self.__class__.__name__,
            cwd=cwd,
            upstream=upstream_ref,
        )
        # Always record raw_argv FIRST so even an argparse-style failure
        # leaves the manifest self-describing.
        manifest.inputs["raw_argv"] = argv_list

        ctx = NodeContext(
            cwd=cwd,
            outputs_dir=outputs_dir,
            manifest_path=manifest_path,
            raw_argv=argv_list,
            config={},
            upstream_pointer=None,
            upstream_manifest=None,
            manifest=manifest,
            started_at_unix=started_at_unix,
            started_at_perf=started_at_perf,
            fail_policy="soft",
        )

        # ---- argv parsing ----
        try:
            if self.accepts_upstream:
                if len(argv_list) < 2:
                    if self.requires_upstream:
                        raise ValueError(
                            "Missing argv[1]: upstream pointer JSON (wf.pointer.v1) is required"
                        )
                    config_args = argv_list[1:]
                else:
                    pointer_text = argv_list[1]
                    upstream_pointer = load_pointer(pointer_text)
                    upstream_ref = UpstreamRef(
                        pointer_schema=upstream_pointer.schema,
                        ok=upstream_pointer.ok,
                        manifest_path=upstream_pointer.manifest_path,
                    )
                    manifest.upstream = upstream_ref.to_dict()
                    ctx.upstream_pointer = upstream_pointer
                    config_args = argv_list[2:]

                    upm_path = Path(upstream_pointer.manifest_path)
                    if upm_path.exists():
                        try:
                            upstream_manifest = Manifest.read(upm_path)
                            ctx.upstream_manifest = upstream_manifest
                        except Exception as e:
                            ctx.fail(f"upstream_manifest_unreadable: {e}")
                    else:
                        ctx.fail(
                            f"upstream_manifest_not_found: {upstream_pointer.manifest_path}"
                        )
            else:
                # Source node: argv[1:] is just config tokens (or e.g. the
                # SMILES string for smiles_to_3d, treated as config).
                config_args = argv_list[1:]

            raw_cfg = parse_kv_or_json(config_args) if config_args else {}

            # fail_policy can be set by the caller via config (read off the
            # raw dict so it works even if a buggy parse_config strips it).
            if isinstance(raw_cfg, dict) and "fail_policy" in raw_cfg:
                fp = str(raw_cfg.get("fail_policy", "soft")).strip().lower()
                ctx.fail_policy = fp if fp in {"soft", "hard"} else "soft"

            # ---- subclass parse hook ----
            typed_cfg = self.parse_config(raw_cfg)
            if not isinstance(typed_cfg, dict):
                raise TypeError(
                    f"{type(self).__name__}.parse_config must return a dict, got {type(typed_cfg).__name__}"
                )

            # The typed cfg is what ``run`` should see — the raw dict has
            # string-typed numerics and is only useful for argv echoing.
            ctx.config = typed_cfg

        except PointerError as e:
            ctx.fail(f"bad_pointer: {e}")
        except Exception as e:
            ctx.fail(f"argv_parse_failed: {e}")
            log_error(traceback.format_exc())

        # ---- run() ----
        # Even if argv parsing failed, we still call run() ONLY if the
        # subclass tolerates that (most don't). The cleanest contract: if
        # ok is already False here due to argv parsing, we skip run() and
        # go straight to finalize, so a guaranteed-manifest is still written.
        if manifest.ok:
            try:
                self.run(ctx)
            except Exception as e:
                ctx.fail(f"run_failed: {e}")
                log_error(traceback.format_exc())

        # ---- finalize ----
        manifest.runtime_seconds = float(time.perf_counter() - started_at_perf)
        manifest.created_at_unix = int(time.time())
        # Merge instead of overwrite: nodes can record their own environment
        # facts (e.g. resolved binary paths, thread counts) during run(), and
        # those should survive finalize. The framework-owned keys
        # (python/python_exe/platform/host) take precedence on conflict so a
        # buggy node can't lie about the interpreter it ran under.
        manifest.environment = {
            **manifest.environment,
            **_detect_environment().to_dict(),
        }

        try:
            manifest.write(manifest_path)
        except Exception as e:
            # Last-ditch: if we cannot even write the manifest, dump a tiny
            # fallback to stderr so the user at least sees something.
            log_error(f"manifest_write_failed: {e}")

        # ---- pointer line ----
        pointer_out = Pointer.of(ok=manifest.ok, manifest_path=manifest_path)
        sys.stdout.write(pointer_out.to_json_line() + "\n")
        sys.stdout.flush()

        if ctx.fail_policy == "hard" and not manifest.ok:
            return 1
        return 0

    # ---------- factory ----------

    @classmethod
    def invoke_factory(cls) -> "Any":
        """Build a callable suitable for use as a console-script entrypoint.

        Usage::

            class MyNode(Node):
                step = "my_step"
                def run(self, ctx): ...

            main = MyNode.invoke_factory()

            if __name__ == "__main__":
                raise SystemExit(main())
        """
        def main() -> int:
            return cls().invoke()
        # Keep a stable __name__ so console-script wheels see ``main``.
        main.__name__ = "main"
        main.__qualname__ = f"{cls.__name__}.main"
        return main


# -----------------------------------------------------------------
# Internals
# -----------------------------------------------------------------


def _detect_environment() -> EnvironmentInfo:
    """Snapshot runtime env metadata for the manifest."""
    try:
        host = socket.gethostname()
    except Exception:
        host = None
    return EnvironmentInfo(
        python=sys.version.split()[0],
        python_exe=str(sys.executable),
        platform=_platform.platform(),
        host=host,
    )
