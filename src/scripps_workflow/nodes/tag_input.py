"""``wf-tag-input`` console-script entrypoint.

This is the thinnest possible shim around :func:`scripps_workflow.tag.tag_main`
so the per-tag-instance ``script.sh`` wrappers can do::

    exec "$ENV_PY" -I -m scripps_workflow.nodes.tag_input <KEY> "$@"

(or, equivalently, call the ``wf-tag-input`` console script with the same
positional args). The actual logic and KEY validation live in
:mod:`scripps_workflow.tag`; everything here is wiring.

A single tag-node instance therefore consists of:

    * one ``script.sh`` produced by :func:`scripps_workflow.env.render_wrapper`
      with ``entrypoint_module="scripps_workflow.nodes.tag_input"`` and
      ``fixed_args=("<KEY>",)``;
    * a one-line ``node.yaml`` declaring inputs/outputs.

No more per-instance ``script.py`` with a hand-edited ``KEY`` constant —
which is exactly the file class that produced the leading-space typo bug.
"""

from __future__ import annotations

from ..tag import tag_main


def main() -> int:
    return tag_main()


if __name__ == "__main__":
    raise SystemExit(main())
