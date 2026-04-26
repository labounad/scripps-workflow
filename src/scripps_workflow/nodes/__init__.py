"""Concrete node implementations.

Each module in this package implements a single workflow node. Per-node
``main`` functions are wired into ``wf-*`` console scripts via
``pyproject.toml`` so the engine's ``script.sh`` is just an exec of one of
those entrypoints.

Modules in this package import lazily — tools like RDKit, prism-pruner,
ORCA-output regexes, etc. are imported inside the node's ``run`` method,
not at module top level, so importing
``scripps_workflow.nodes.smiles_to_3d`` on a machine without RDKit does not
explode (it just records a soft failure when the node is actually
invoked).
"""
