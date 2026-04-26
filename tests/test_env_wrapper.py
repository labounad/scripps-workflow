"""Tests for the strict env-guard wrapper template.

The wrapper is the canonical fix for the Franken-Python contamination
problem and the per-node copy-paste sprawl. These tests pin its behavior:
strict guard variables, module-load support, correct exec.
"""

from __future__ import annotations

import stat

from scripps_workflow.env import (
    DEFAULT_ENV_PY,
    PRISM_ENV_PY,
    render_wrapper,
    write_wrapper,
)


class TestRenderWrapper:
    def test_default_wrapper_unsets_python_vars(self):
        text = render_wrapper(entrypoint_module="scripps_workflow.nodes.smiles_to_3d")
        assert "unset PYTHONHOME PYTHONPATH PYTHONSTARTUP" in text
        # Critically: PYTHONNOUSERSITE must be EXPORTED (not unset like the
        # regressed newer wrappers).
        assert "export PYTHONNOUSERSITE=1" in text
        assert "unset PYTHONNOUSERSITE" not in text

    def test_default_wrapper_uses_strict_isolation(self):
        text = render_wrapper(entrypoint_module="scripps_workflow.nodes.smiles_to_3d")
        # `-I` (full isolation), not `-E` (less strict).
        assert " -I " in text
        assert text.count(" -E ") == 0

    def test_uses_python_dash_m(self):
        text = render_wrapper(entrypoint_module="scripps_workflow.nodes.smiles_to_3d")
        assert "-m scripps_workflow.nodes.smiles_to_3d" in text

    def test_default_env_py_is_group_workflow_env(self):
        text = render_wrapper(entrypoint_module="scripps_workflow.nodes.xtb_calc")
        assert DEFAULT_ENV_PY in text

    def test_env_py_overridable_for_prism(self):
        text = render_wrapper(
            entrypoint_module="scripps_workflow.nodes.prism_screen",
            env_py=PRISM_ENV_PY,
        )
        assert PRISM_ENV_PY in text

    def test_env_py_can_be_runtime_overridden_by_caller(self):
        # The wrapper template uses ${ENV_PY:-default} so a caller can
        # `ENV_PY=/x/python script.sh` to override.
        text = render_wrapper(entrypoint_module="scripps_workflow.nodes.smiles_to_3d")
        assert 'ENV_PY="${ENV_PY:-' in text


class TestModuleLoads:
    def test_no_module_loads_means_no_module_block(self):
        text = render_wrapper(entrypoint_module="scripps_workflow.nodes.x")
        assert "module load" not in text

    def test_module_loads_inserted_in_order(self):
        text = render_wrapper(
            entrypoint_module="scripps_workflow.nodes.orca_dft_array",
            module_loads=("orca/6.1.0", "openmpi/4.1"),
        )
        assert "module load orca/6.1.0" in text
        assert "module load openmpi/4.1" in text
        assert text.index("orca/6.1.0") < text.index("openmpi/4.1")

    def test_module_loads_source_modules_init(self):
        # Required so `module` works in non-interactive SLURM shells.
        text = render_wrapper(
            entrypoint_module="scripps_workflow.nodes.xtb_calc",
            module_loads=("xtb/6.6.1",),
        )
        assert "/etc/profile.d/modules.sh" in text


class TestFixedArgs:
    def test_no_fixed_args_means_just_dollar_at(self):
        # No fixed_args → ``-m <module> "$@"`` with nothing in between.
        text = render_wrapper(entrypoint_module="scripps_workflow.nodes.tag_input")
        assert '-m scripps_workflow.nodes.tag_input "$@"' in text

    def test_single_fixed_arg_is_baked_in(self):
        # Tag-node use case: ``-m tag_input temperature_k "$@"``.
        text = render_wrapper(
            entrypoint_module="scripps_workflow.nodes.tag_input",
            fixed_args=("temperature_k",),
        )
        assert (
            '-m scripps_workflow.nodes.tag_input temperature_k "$@"' in text
        )

    def test_multiple_fixed_args_in_order(self):
        text = render_wrapper(
            entrypoint_module="scripps_workflow.nodes.tag_input",
            fixed_args=("foo", "bar"),
        )
        assert (
            '-m scripps_workflow.nodes.tag_input foo bar "$@"' in text
        )

    def test_fixed_arg_with_spaces_is_quoted(self):
        # shlex.quote-protected so a malicious or accidental embedded
        # space cannot inject extra shell tokens.
        text = render_wrapper(
            entrypoint_module="scripps_workflow.nodes.tag_input",
            fixed_args=("not a real key",),
        )
        # Should appear quoted, not as three bare tokens.
        assert "'not a real key'" in text
        assert " not a real key " not in text

    def test_fixed_arg_with_shell_metacharacter_is_quoted(self):
        text = render_wrapper(
            entrypoint_module="scripps_workflow.nodes.tag_input",
            fixed_args=("$(rm -rf /)",),
        )
        # Must be single-quoted so the shell does NOT execute it.
        assert "'$(rm -rf /)'" in text


class TestWriteWrapper:
    def test_write_makes_executable(self, tmp_path):
        target = tmp_path / "script.sh"
        write_wrapper(target, entrypoint_module="scripps_workflow.nodes.smiles_to_3d")
        st = target.stat()
        # Owner should at least have execute permission.
        assert st.st_mode & stat.S_IXUSR

    def test_write_creates_parent_dirs(self, tmp_path):
        target = tmp_path / "deep" / "nested" / "script.sh"
        write_wrapper(target, entrypoint_module="scripps_workflow.nodes.smiles_to_3d")
        assert target.exists()

    def test_written_text_matches_render(self, tmp_path):
        target = tmp_path / "script.sh"
        write_wrapper(target, entrypoint_module="scripps_workflow.nodes.smiles_to_3d")
        expected = render_wrapper(entrypoint_module="scripps_workflow.nodes.smiles_to_3d")
        assert target.read_text(encoding="utf-8") == expected
