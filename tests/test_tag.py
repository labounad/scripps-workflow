"""Tests for the universal tag-node helper.

Covers the contract:
    * Valid KEY + value emits ``KEY=VALUE\\n`` exactly to stdout.
    * Invalid KEY (whitespace / empty / illegal chars) emits a
      ``tagger_error=<reason>`` token, never a malformed key.
    * All paths return exit code 0 (soft-fail invariant).
    * Multi-token values are joined with a single space.
    * The ``wf-tag-input`` console-script entrypoint is a thin shim.
"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stderr, redirect_stdout

import pytest

from scripps_workflow.nodes.tag_input import main as tag_input_main
from scripps_workflow.tag import tag_main


def _run(argv: list[str]) -> tuple[int, str, str]:
    """Run ``tag_main(argv)`` capturing stdout, stderr, and exit code."""
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = tag_main(argv)
    return rc, out.getvalue(), err.getvalue()


class TestHappyPath:
    def test_simple_key_value(self) -> None:
        rc, out, _ = _run(["tag_input", "temperature_k", "298.15"])
        assert rc == 0
        assert out == "temperature_k=298.15\n"

    def test_value_with_decimal(self) -> None:
        rc, out, _ = _run(["tag_input", "ewin_kcal", "5.0"])
        assert rc == 0
        assert out == "ewin_kcal=5.0\n"

    def test_value_is_json_blob_single_token(self) -> None:
        rc, out, _ = _run(
            ["tag_input", "config", '{"recompute_low": false}']
        )
        assert rc == 0
        assert out == 'config={"recompute_low": false}\n'

    def test_kebab_case_key(self) -> None:
        rc, out, _ = _run(["tag_input", "max-concurrency", "10"])
        assert rc == 0
        assert out == "max-concurrency=10\n"

    def test_underscore_only_key(self) -> None:
        rc, out, _ = _run(["tag_input", "_private_key", "ok"])
        assert rc == 0
        assert out == "_private_key=ok\n"


class TestMissingValue:
    def test_missing_value_emits_empty(self) -> None:
        # Matches legacy tag_temperature_k.py behavior: emit ``KEY=`` so
        # the downstream parser sees a well-formed (if empty) token.
        rc, out, err = _run(["tag_input", "calculations"])
        assert rc == 0
        assert out == "calculations=\n"
        assert "missing value" in err.lower()


class TestInvalidKey:
    def test_leading_space_is_rejected(self) -> None:
        # THIS is the load-bearing test — the legacy bug was
        # ``KEY = " max_concurrency"`` producing a token that downstream
        # parsers turned into a dict key with a leading space, which
        # silently missed every consumer's lookup.
        rc, out, err = _run(["tag_input", " max_concurrency", "10"])
        assert rc == 0
        assert out == "tagger_error=whitespace_in_key\n"
        assert "whitespace" in err.lower()

    def test_trailing_space_is_rejected(self) -> None:
        rc, out, _ = _run(["tag_input", "temperature_k ", "298"])
        assert rc == 0
        assert out == "tagger_error=whitespace_in_key\n"

    def test_empty_key_is_rejected(self) -> None:
        rc, out, _ = _run(["tag_input", "", "298"])
        assert rc == 0
        assert out == "tagger_error=empty_key\n"

    def test_whitespace_only_key_is_rejected(self) -> None:
        rc, out, _ = _run(["tag_input", "   ", "298"])
        # Stripping leaves empty → empty_key, not whitespace_in_key.
        # Either is acceptable; we just want any tagger_error and rc=0.
        assert rc == 0
        assert out.startswith("tagger_error=")
        assert out.endswith("\n")

    def test_key_starting_with_digit_is_rejected(self) -> None:
        rc, out, _ = _run(["tag_input", "1bad", "x"])
        assert rc == 0
        assert out == "tagger_error=invalid_key\n"

    def test_key_with_special_chars_is_rejected(self) -> None:
        rc, out, _ = _run(["tag_input", "bad.key", "x"])
        assert rc == 0
        assert out == "tagger_error=invalid_key\n"

    def test_key_with_equals_in_it_is_rejected(self) -> None:
        rc, out, _ = _run(["tag_input", "key=oops", "x"])
        assert rc == 0
        assert out == "tagger_error=invalid_key\n"


class TestMissingKey:
    def test_no_key_argument_at_all(self) -> None:
        rc, out, err = _run(["tag_input"])
        assert rc == 0
        assert out == "tagger_error=missing_key\n"
        assert "missing key" in err.lower()


class TestMultiTokenValue:
    def test_two_tokens_joined_with_space(self) -> None:
        rc, out, _ = _run(["tag_input", "note", "hello", "world"])
        assert rc == 0
        assert out == "note=hello world\n"

    def test_three_tokens_joined(self) -> None:
        rc, out, _ = _run(["tag_input", "note", "a", "b", "c"])
        assert rc == 0
        assert out == "note=a b c\n"


class TestStdoutContract:
    def test_exactly_one_line_with_trailing_newline(self) -> None:
        _, out, _ = _run(["tag_input", "k", "v"])
        # One line, ends with exactly one newline, no extra blank lines.
        assert out.count("\n") == 1
        assert out.endswith("\n")

    def test_no_stdout_pollution_from_logs(self) -> None:
        # Errors must go to stderr, never stdout. A noisy tag node would
        # break engine wiring (downstream argv would include stray text).
        _, out, err = _run(["tag_input", " bad", "v"])
        assert out.count("\n") == 1  # exactly the tagger_error token
        assert err  # we did log something


class TestSoftFailInvariant:
    @pytest.mark.parametrize(
        "argv",
        [
            ["tag_input"],
            ["tag_input", ""],
            ["tag_input", " bad", "v"],
            ["tag_input", "bad.key", "v"],
            ["tag_input", "ok"],
            ["tag_input", "ok", "v"],
        ],
    )
    def test_always_returns_zero(self, argv: list[str]) -> None:
        rc, _, _ = _run(argv)
        assert rc == 0


class TestConsoleScriptShim:
    def test_shim_delegates_to_tag_main(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The console-script entrypoint is a one-liner; verify it really
        # does call tag_main with the live sys.argv (i.e. the wrapper
        # script's positional args reach the helper unmodified).
        monkeypatch.setattr(sys, "argv", ["wf-tag-input", "k", "v"])
        out = io.StringIO()
        with redirect_stdout(out):
            rc = tag_input_main()
        assert rc == 0
        assert out.getvalue() == "k=v\n"
