"""scripps_workflow — engine-agnostic node + workflow framework.

This package owns the schema, base classes, and helpers that workflow nodes
share. The engine (workflow.scripps.edu) is a *consumer* of node definitions
exported from this repo; it is not the source of truth.

Runtime layer (this module + .pointer, .schema, .parsing, .hashing, .env,
.logging_utils, .node) is **stdlib-only**: no third-party imports. That keeps
node scripts portable across HPC envs.

GUI round-trip tooling (.gui.*) requires the ``[gui]`` extra (pydantic, pyyaml)
and is loaded only by `wf-export-gui` / `wf-import-gui` / `wf-validate`.

Per-node logic lives in ``scripps_workflow.nodes.*`` and is invoked via the
``wf-*`` console scripts declared in ``pyproject.toml``.
"""

from __future__ import annotations

__version__ = "0.0.1"

# Re-exports for the most common imports inside node code. Anything pulled in
# here MUST stay stdlib-only.
from .pointer import (
    POINTER_SCHEMA,
    Pointer,
    dump_pointer,
    load_pointer,
)
from .schema import (
    RESULT_SCHEMA,
    ArtifactRecord,
    EnvironmentInfo,
    Manifest,
    UpstreamRef,
)
from .parsing import (
    normalize_optional_str,
    normalize_tri,
    parse_bool,
    parse_float,
    parse_int,
    parse_kv_or_json,
    parse_optional_int,
)
from .hashing import sha256_file
from .logging_utils import log, log_error, log_info, log_warn
from .node import Node, NodeContext
from .tag import tag_main

__all__ = [
    "__version__",
    # pointer
    "POINTER_SCHEMA",
    "Pointer",
    "dump_pointer",
    "load_pointer",
    # schema
    "RESULT_SCHEMA",
    "ArtifactRecord",
    "EnvironmentInfo",
    "Manifest",
    "UpstreamRef",
    # parsing
    "normalize_optional_str",
    "normalize_tri",
    "parse_bool",
    "parse_float",
    "parse_int",
    "parse_kv_or_json",
    "parse_optional_int",
    # hashing
    "sha256_file",
    # logging
    "log",
    "log_error",
    "log_info",
    "log_warn",
    # node base
    "Node",
    "NodeContext",
    # tag helper
    "tag_main",
]
