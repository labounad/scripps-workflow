"""Standalone output-viewer bundle builders.

The workflow GUI's live Output-node inport resolver is not reliable enough for
chemistry viewers that need to load HPC-produced files in an iframe.  The
modules in this package instead build downloadable, self-contained ZIP bundles
that contain an ``index.html`` viewer plus embedded XYZ payload data.

The GUI node bundles are intentionally tiny shims that import these modules
from the installed ``scripps-workflow`` package.  That keeps iteration fast:
fix viewer behavior in the repo, pull/install on the HPC, and rerun without
re-importing GUI nodes unless the GUI-facing input/output contract changes.
"""

from __future__ import annotations

__all__ = []
