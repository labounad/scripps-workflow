"""Load standalone molecular viewer assets from package static files.

The viewer HTML/CSS/JS are intentionally stored as ordinary frontend files
under ``output_viewers/static`` instead of as giant Python string literals.
``bundle_common.write_bundle`` still consumes strings, so this module keeps
the historical constants while delegating the actual source of truth to the
static asset tree.
"""

from __future__ import annotations

from importlib.resources import files


_STATIC_ROOT = files(__package__) / "static"


def _read_text(relative_path: str) -> str:
    return (_STATIC_ROOT / relative_path).read_text(encoding="utf-8")


def _concat_text(relative_dir: str) -> str:
    directory = _STATIC_ROOT / relative_dir
    return "\n\n".join(
        path.read_text(encoding="utf-8").rstrip()
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix in {".js", ".css", ".html"}
    ) + "\n"


ENSEMBLE_INDEX_HTML = _read_text("ensemble/index.html")
ENSEMBLE_VIEWER_JS = _concat_text("ensemble/js")

GEOMETRY_INDEX_HTML = _read_text("geometry/index.html")
GEOMETRY_VIEWER_JS = _read_text("geometry/viewer.js")

COMMON_CSS = _read_text("common/styles.css")

# Backward-compatible aliases used by older bundle builders/tests.
ENSEMBLE_CSS = COMMON_CSS
GEOMETRY_CSS = COMMON_CSS
