"""Shared configuration for generated GUI node bundles.

The workflow GUI executes Output/Layout node scripts in a very constrained
runtime.  Keep the hard-won bootstrap defaults in one place so future HPC path
or author-metadata changes are not copied across every generator.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

OUT_DIR = Path("new_nodes_output")
PLACEHOLDER_NODE_ID = 999
NODE_VERSION = "1.2.4"
WORKFLOW_HOST = "workflow.scripps.edu"
DEFAULT_WORKFLOW_PYTHON = "/gpfs/group/shenvi/envs/workflow312/bin/python"
DEFAULT_REPO_SRC_CANDIDATES = (
    "/gpfs/group/shenvi/Users/labounader/scripps-workflow/src",
    "/gpfs/home/labounader/scripps-workflow/src",
)

AUTHOR = {
    "user_id": 102,
    "name": "Lucas",
    "lastname": "Abounader",
    "lab": "Shenvi",
    "email": "labounader@scripps.edu",
    "main_role": "Designer",
    "first_connection": "2026-02-13 21:09:00",
    "last_connection": "2026-05-16 14:30:00",
    "isAzureUser": 0,
}


@dataclass(frozen=True)
class InputSpec:
    """Declarative GUI input definition used by output-node generators."""

    name: str
    type: str = "text"
    required: int = 0
    tags: str = ""
