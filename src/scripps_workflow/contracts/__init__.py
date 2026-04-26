"""Role-contract validators.

Each module here corresponds to a role declared in ``nodes/roles/<role>.yaml``
and exposes a stdlib-only ``validate(manifest_dict)`` function returning
a list of human-readable problem strings. Empty list = the manifest
satisfies the contract.

Validators live here (not in ``schema.py``) because ``schema.py`` describes
the *generic* manifest envelope — every node satisfies it. Roles add
*specific* requirements on top: a ``conformer_screen`` impl must have an
``accepted`` bucket whose elements have a ``sha256`` field, etc. Keeping
those requirements in their own module preserves the layering: a node
that doesn't fill any role ignores ``contracts.*`` entirely.

Pydantic models for the same contracts live under the ``[gui]`` extra in
``scripps_workflow.gui.contracts.*`` (not yet implemented). The two are
kept in sync by hand for now; once the GUI tooling lands, the Pydantic
models will be the schema source-of-truth and these stdlib validators
will become a thin adapter.
"""

from __future__ import annotations

from .conformer_screen import (
    CONFORMER_SCREEN_VERSION,
    validate_conformer_screen,
)

__all__ = [
    "CONFORMER_SCREEN_VERSION",
    "validate_conformer_screen",
]
