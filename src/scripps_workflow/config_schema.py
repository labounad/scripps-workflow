"""Declarative config-schema framework for ``Node`` subclasses.

Each node ships a :class:`NodeSchema` listing its config knobs as
:class:`ConfigField` entries. A single :func:`apply_schema` helper
replaces hand-rolled ``parse_config`` bodies: it reads each field
from the raw argv dict, coerces types, applies validators, and
returns a typed cfg dict the node's ``run`` method can consume
unchanged.

The schema is also the source of truth for auto-generated docs
(``tools/gen_config_docs.py``) and, eventually, GUI input slot
JSONs — eliminating the GUI-vs-Python drift bug we caught when
``thermo_aggregate``'s ``n_tasks_override`` slot didn't match the
Python's ``n_tasks`` read.

Design:

* ``ConfigField`` is the unit of declaration. Mandatory: ``name``,
  ``type``, ``description``. Optional everything else.
* ``type`` is a short string token (``"str"`` / ``"int"`` /
  ``"float"`` / ``"bool"`` / ``"csv"`` / ``"json"`` / ``"enum"``)
  rather than a Python type object — keeps the docs human-readable
  and the GUI generator simple.
* ``default`` of :data:`REQUIRED` makes the field mandatory.
* ``aliases`` lets older raw-argv keys keep working (e.g. ``wf-prism``
  accepts ``energy_window_kcal`` as an alias for ``max_dE_kcal``).
* ``choices`` enumerates valid values for ``type="enum"``.
* ``min_value`` / ``max_value`` range-check numeric fields.
* ``coercer`` overrides standard type coercion when a field needs
  unusual parsing (e.g., ``mnova_heteronuclear_partners`` validates
  against a whitelist of element symbols).
* ``validator`` runs after coercion for per-field invariants that
  aren't expressible as min/max (e.g., "must be a basename, no
  slashes").
* ``depends_on`` is doc-only — names other fields this one interacts
  with, surfaced in the generated reference.
* ``section`` groups related fields under a sub-heading in the
  rendered docs (e.g., ``"mnova"`` vs ``"diagrams"`` vs ``"slurm"``).
"""

from __future__ import annotations

import json as _json
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


#: Sentinel for fields with no default (must be provided by the caller).
REQUIRED: Any = object()


@dataclass(frozen=True)
class ConfigField:
    """Declarative spec for one config knob.

    The dataclass is frozen so a schema object is hashable and
    accidentally mutating a field after declaration raises. Use
    :func:`dataclasses.replace` to create a tweaked variant.
    """

    name: str
    type: str
    description: str = ""
    default: Any = REQUIRED
    aliases: tuple[str, ...] = ()
    choices: Optional[tuple[Any, ...]] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    coercer: Optional[Callable[[Any], Any]] = None
    validator: Optional[Callable[[Any], Any]] = None
    depends_on: tuple[str, ...] = ()
    section: str = ""

    def is_required(self) -> bool:
        return self.default is REQUIRED


@dataclass(frozen=True)
class NodeSchema:
    """Config schema for one node — the source of truth for parsing + docs."""

    step_name: str
    cli_entrypoint: str
    module_path: str
    overview: str = ""
    fields: tuple[ConfigField, ...] = ()

    def field_by_name(self, name: str) -> Optional[ConfigField]:
        for f in self.fields:
            if f.name == name or name in f.aliases:
                return f
        return None


# --------------------------------------------------------------------
# Type coercion
# --------------------------------------------------------------------


def _coerce_str(value: Any) -> str:
    return str(value).strip()


def _coerce_int(value: Any) -> int:
    if isinstance(value, bool):  # bool is an int subclass; reject silently-bad coercion
        return int(value)
    if isinstance(value, int):
        return value
    return int(str(value).strip())


def _coerce_float(value: Any) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return float(str(value).strip())


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    s = str(value).strip().lower()
    if s in {"true", "yes", "y", "t", "1", "on"}:
        return True
    if s in {"false", "no", "n", "f", "0", "off", ""}:
        return False
    raise ValueError(f"cannot coerce {value!r} to bool")


def _coerce_csv(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(x).strip() for x in value if str(x).strip()]
    return [s.strip() for s in str(value).split(",") if s.strip()]


def _coerce_json(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if value is None or value == "":
        return None
    return _json.loads(value) if isinstance(value, str) else value


_COERCERS: dict[str, Callable[[Any], Any]] = {
    "str": _coerce_str,
    "int": _coerce_int,
    "float": _coerce_float,
    "bool": _coerce_bool,
    "csv": _coerce_csv,
    "json": _coerce_json,
    "enum": _coerce_str,
}


# --------------------------------------------------------------------
# apply_schema — the parse_config replacement
# --------------------------------------------------------------------


def apply_schema(raw: dict[str, Any], schema: NodeSchema) -> dict[str, Any]:
    """Parse raw config dict into a typed cfg dict using ``schema``.

    For each :class:`ConfigField`:

    1. Look up the raw value by ``name`` (or any ``aliases``). If not
       present, use ``default`` — or raise ``ValueError`` when the
       field is :data:`REQUIRED`.
    2. Coerce the value using the field's ``coercer`` if present,
       otherwise the standard coercer for ``type``.
    3. Validate ``choices`` (for ``type="enum"``) and ``min_value``
       / ``max_value`` (for numeric types).
    4. Run the field's ``validator`` callable if present; its return
       value replaces the coerced value (so validators can normalize
       as well as check).

    Returns a flat dict keyed by canonical field names. Aliases are
    NOT preserved in the output — the node's ``run`` method always
    sees the canonical key.

    Raises ``ValueError`` on the first invalid field, with a message
    naming the offending key. The framework's surrounding ``Node``
    machinery catches that and surfaces ``argv_parse_failed`` so the
    failure shows up in the manifest rather than crashing the run.
    """
    cfg: dict[str, Any] = {}
    for fld in schema.fields:
        # 1. Resolve from raw via name + aliases.
        value: Any = None
        present = False
        for key in (fld.name, *fld.aliases):
            if key in raw and raw[key] is not None and raw[key] != "":
                value = raw[key]
                present = True
                break

        if not present:
            if fld.is_required():
                raise ValueError(f"{fld.name}: required, not provided")
            cfg[fld.name] = fld.default
            continue

        # 2. Coerce.
        coercer = fld.coercer or _COERCERS.get(fld.type)
        if coercer is None:
            raise ValueError(
                f"{fld.name}: unknown type {fld.type!r}; "
                f"expected one of {sorted(_COERCERS)}"
            )
        try:
            coerced = coercer(value)
        except (ValueError, TypeError) as e:
            raise ValueError(f"{fld.name}: {e}") from e

        # 3. Enum / range validation.
        if fld.type == "enum" and fld.choices is not None:
            if coerced not in fld.choices:
                raise ValueError(
                    f"{fld.name}: must be one of {list(fld.choices)}, "
                    f"got {coerced!r}"
                )
        if isinstance(coerced, (int, float)) and not isinstance(coerced, bool):
            if fld.min_value is not None and coerced < fld.min_value:
                raise ValueError(
                    f"{fld.name}: must be >= {fld.min_value}, got {coerced}"
                )
            if fld.max_value is not None and coerced > fld.max_value:
                raise ValueError(
                    f"{fld.name}: must be <= {fld.max_value}, got {coerced}"
                )

        # 4. Custom validator (can normalize + check).
        if fld.validator is not None:
            try:
                coerced = fld.validator(coerced)
            except (ValueError, TypeError) as e:
                raise ValueError(f"{fld.name}: {e}") from e

        cfg[fld.name] = coerced
    return cfg


__all__ = [
    "ConfigField",
    "NodeSchema",
    "REQUIRED",
    "apply_schema",
]
