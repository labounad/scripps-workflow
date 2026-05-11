"""Generate ``docs/CONFIG_REFERENCE.md`` from per-node :class:`NodeSchema` objects.

Walks every module in ``scripps_workflow.nodes``, imports it, and
collects any module-level ``SCHEMA: NodeSchema`` attribute. Renders
a single markdown file with TOC + per-node section.

The output is fully deterministic: same set of schemas → identical
bytes. Wire this into a pre-commit hook or CI check to detect when
docs drift from code:

    python tools/gen_config_docs.py --check

exits non-zero when ``docs/CONFIG_REFERENCE.md`` doesn't match what
the script would generate fresh. Without ``--check`` it writes the
file and exits 0.

Nodes that don't yet have a SCHEMA are listed in a "Not yet ported"
section at the bottom — a reminder of remaining migration work.
"""

from __future__ import annotations

import argparse
import importlib
import pkgutil
import sys
from pathlib import Path

# Ensure src/ is on the import path when run from the repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from scripps_workflow.config_schema import (  # noqa: E402
    REQUIRED,
    ConfigField,
    NodeSchema,
)


# --------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------


def _discover_schemas() -> tuple[list[NodeSchema], list[str]]:
    """Walk ``scripps_workflow.nodes`` and return (schemas, unported_modules).

    Returns:
        schemas: ordered by step_name for stable output.
        unported_modules: short module names that imported successfully
            but didn't expose a SCHEMA attribute. Listed in the docs
            as a migration TODO.
    """
    import scripps_workflow.nodes as nodes_pkg

    schemas: list[NodeSchema] = []
    unported: list[str] = []
    for info in pkgutil.iter_modules(nodes_pkg.__path__):
        modname = f"scripps_workflow.nodes.{info.name}"
        try:
            mod = importlib.import_module(modname)
        except Exception as e:
            unported.append(f"{info.name} (import failed: {e})")
            continue
        schema = getattr(mod, "SCHEMA", None)
        if isinstance(schema, NodeSchema):
            schemas.append(schema)
        else:
            unported.append(info.name)

    schemas.sort(key=lambda s: s.step_name)
    unported.sort()
    return schemas, unported


# --------------------------------------------------------------------
# Markdown rendering
# --------------------------------------------------------------------


def _slug(s: str) -> str:
    """GitHub-style heading slug. Lowercases, replaces non-alphanum
    with hyphens, collapses consecutive hyphens. ``wf-nmr-aggregate``
    stays ``wf-nmr-aggregate``."""
    out: list[str] = []
    for ch in s.lower():
        if ch.isalnum():
            out.append(ch)
        elif ch in {"-", "_", " "}:
            out.append("-")
    slug = "".join(out)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-")


def _fmt_default(field: ConfigField) -> str:
    if field.is_required():
        return "**required**"
    d = field.default
    if d is None:
        return "`None`"
    if isinstance(d, str):
        return f"`{d!r}`" if d == "" else f"`\"{d}\"`"
    if isinstance(d, (list, tuple)) and not d:
        return "`[]`"
    return f"`{d!r}`"


def _render_field(field: ConfigField) -> str:
    lines: list[str] = []
    lines.append(f"#### `{field.name}`")
    lines.append("")
    lines.append(f"- **Type:** `{field.type}`")
    lines.append(f"- **Default:** {_fmt_default(field)}")
    if field.aliases:
        aliases = ", ".join(f"`{a}`" for a in field.aliases)
        lines.append(f"- **Aliases:** {aliases}")
    if field.choices:
        choices = ", ".join(f"`{c}`" for c in field.choices)
        lines.append(f"- **Choices:** {choices}")
    if field.min_value is not None:
        lines.append(f"- **Min:** `{field.min_value}`")
    if field.max_value is not None:
        lines.append(f"- **Max:** `{field.max_value}`")
    if field.depends_on:
        related = ", ".join(f"`{d}`" for d in field.depends_on)
        lines.append(f"- **Related:** {related}")
    lines.append("")
    if field.description:
        lines.append(field.description.strip())
        lines.append("")
    return "\n".join(lines)


def _render_node(schema: NodeSchema) -> str:
    lines: list[str] = []
    lines.append(f"## `{schema.cli_entrypoint}`")
    lines.append("")
    lines.append(f"- **Step:** `{schema.step_name}`")
    lines.append(f"- **Module:** `{schema.module_path}`")
    lines.append("")
    if schema.overview:
        lines.append(schema.overview.strip())
        lines.append("")
    lines.append("### Config keys")
    lines.append("")

    # Group fields by section. Stable order: first-seen-section
    # ordering, with un-sectioned ("") fields rendered without a
    # subheading.
    section_order: list[str] = []
    by_section: dict[str, list[ConfigField]] = {}
    for fld in schema.fields:
        sec = fld.section or ""
        if sec not in by_section:
            section_order.append(sec)
            by_section[sec] = []
        by_section[sec].append(fld)

    for sec in section_order:
        if sec:
            lines.append(f"#### Section: {sec}")
            lines.append("")
        for fld in by_section[sec]:
            lines.append(_render_field(fld))
    return "\n".join(lines)


def _render_doc(
    schemas: list[NodeSchema], unported: list[str]
) -> str:
    parts: list[str] = []
    parts.append("# scripps-workflow config reference")
    parts.append("")
    parts.append(
        "Auto-generated by `tools/gen_config_docs.py` from per-node "
        "`NodeSchema` declarations. Do not hand-edit — changes will "
        "be overwritten on the next regeneration. To update an entry, "
        "edit the `SCHEMA` block in the corresponding node module."
    )
    parts.append("")

    # TOC
    parts.append("## Nodes")
    parts.append("")
    for s in schemas:
        parts.append(f"- [`{s.cli_entrypoint}`](#{_slug(s.cli_entrypoint)})")
    parts.append("")

    # Per-node sections
    for s in schemas:
        parts.append(_render_node(s))

    # Unported tail
    if unported:
        parts.append("## Not yet ported to schema")
        parts.append("")
        parts.append(
            "These node modules don't yet declare a module-level "
            "`SCHEMA` — they still use hand-rolled `parse_config` "
            "bodies. Their config knobs are documented in the module "
            "docstrings until migration."
        )
        parts.append("")
        for name in unported:
            parts.append(f"- `{name}`")
        parts.append("")

    return "\n".join(parts) + "\n"


# --------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "Don't write — exit non-zero if the existing file differs "
            "from what would be generated. Use in CI / pre-commit."
        ),
    )
    parser.add_argument(
        "--out",
        default="docs/CONFIG_REFERENCE.md",
        help="Output path (default: docs/CONFIG_REFERENCE.md)",
    )
    args = parser.parse_args()

    schemas, unported = _discover_schemas()
    rendered = _render_doc(schemas, unported)

    out_path = Path(args.out)
    if args.check:
        if not out_path.exists():
            print(f"FAIL: {out_path} does not exist", file=sys.stderr)
            return 1
        existing = out_path.read_text(encoding="utf-8")
        if existing != rendered:
            print(
                f"FAIL: {out_path} is out of date. "
                f"Run `python tools/gen_config_docs.py` to regenerate.",
                file=sys.stderr,
            )
            return 1
        print(f"OK: {out_path} is up to date.")
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(rendered, encoding="utf-8")
    print(
        f"wrote {out_path} "
        f"({len(schemas)} schema(s), {len(unported)} unported)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
