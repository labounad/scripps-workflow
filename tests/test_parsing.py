"""Tests for argv-token parsing helpers."""

from __future__ import annotations

import pytest

from scripps_workflow.parsing import (
    normalize_optional_str,
    normalize_tri,
    parse_bool,
    parse_float,
    parse_int,
    parse_kv_or_json,
    parse_optional_int,
)


class TestParseKvOrJson:
    def test_empty_returns_empty(self):
        assert parse_kv_or_json([]) == {}

    def test_simple_kv(self):
        assert parse_kv_or_json(["a=1", "b=2"]) == {"a": "1", "b": "2"}

    def test_strips_whitespace_around_key_and_value(self):
        # Fixes the leading-space tag-node bug ("KEY = ' max_concurrency'").
        result = parse_kv_or_json([" max_concurrency = 10 "])
        assert result == {"max_concurrency": "10"}

    def test_value_can_contain_equals(self):
        # ORCA keyword strings often have `=`. Only the FIRST `=` splits.
        result = parse_kv_or_json(["keywords=! r2scan-3c TightSCF Freq"])
        assert result == {"keywords": "! r2scan-3c TightSCF Freq"}

    def test_single_json_object_parsed_as_object(self):
        result = parse_kv_or_json(['{"max_dE_kcal": 10, "use_energies": "auto"}'])
        assert result == {"max_dE_kcal": 10, "use_energies": "auto"}

    def test_json_object_must_be_object(self):
        with pytest.raises(ValueError, match="must be an object"):
            parse_kv_or_json(["[1, 2, 3]"])

    def test_token_without_equals_raises(self):
        with pytest.raises(ValueError, match="key=value"):
            parse_kv_or_json(["just_a_word"])

    def test_empty_key_raises(self):
        with pytest.raises(ValueError, match="empty key"):
            parse_kv_or_json(["=value"])

    def test_invalid_key_chars_raises(self):
        with pytest.raises(ValueError, match="alphanumeric"):
            parse_kv_or_json(["weird key!=value"])


class TestParseInt:
    def test_simple(self):
        assert parse_int("10", 0) == 10

    def test_negative(self):
        assert parse_int("-5", 0) == -5

    def test_whitespace_tolerated(self):
        assert parse_int(" 7 ", 0) == 7

    def test_none_returns_default(self):
        assert parse_int(None, 99) == 99

    def test_empty_string_returns_default(self):
        assert parse_int("", 99) == 99

    def test_sentinels_return_default(self):
        for s in ("none", "null", "auto", "NONE", "Auto"):
            assert parse_int(s, 42) == 42

    def test_garbage_returns_default(self):
        assert parse_int("nope", 42) == 42

    def test_float_string_returns_default(self):
        # parse_int is strict — 1.5 is not an int.
        assert parse_int("1.5", 42) == 42


class TestParseFloat:
    def test_simple(self):
        assert parse_float("1.5", 0.0) == 1.5

    def test_int_string_works(self):
        assert parse_float("3", 0.0) == 3.0

    def test_scientific(self):
        assert parse_float("1.0e-3", 0.0) == 1e-3

    def test_sentinels_return_default(self):
        assert parse_float("auto", 0.5) == 0.5
        assert parse_float("none", 0.5) == 0.5


class TestParseBool:
    @pytest.mark.parametrize("v", ["true", "TRUE", "1", "yes", "y", "on", "t", True, 1])
    def test_true_values(self, v):
        assert parse_bool(v, False) is True

    @pytest.mark.parametrize("v", ["false", "FALSE", "0", "no", "n", "off", "f", False, 0])
    def test_false_values(self, v):
        assert parse_bool(v, True) is False

    def test_unrecognized_returns_default(self):
        assert parse_bool("maybe", True) is True
        assert parse_bool("maybe", False) is False

    def test_none_returns_default(self):
        assert parse_bool(None, True) is True


class TestNormalizeTri:
    def test_auto(self):
        assert normalize_tri("auto") == "auto"
        assert normalize_tri(None) == "auto"
        assert normalize_tri("") == "auto"

    def test_true(self):
        assert normalize_tri("true") == "true"
        assert normalize_tri("yes") == "true"
        assert normalize_tri("1") == "true"

    def test_false(self):
        assert normalize_tri("false") == "false"
        assert normalize_tri("0") == "false"

    def test_unrecognized_returns_default(self):
        assert normalize_tri("xyz", default="false") == "false"


class TestParseOptionalInt:
    def test_int(self):
        assert parse_optional_int("5") == 5

    def test_none_returns_none(self):
        assert parse_optional_int(None) is None

    def test_sentinels_return_none(self):
        assert parse_optional_int("auto") is None
        assert parse_optional_int("") is None


class TestNormalizeOptionalStr:
    def test_string_passthrough(self):
        assert normalize_optional_str("hello") == "hello"

    def test_strip_whitespace(self):
        assert normalize_optional_str("  hello  ") == "hello"

    def test_empty_returns_none(self):
        assert normalize_optional_str("") is None
        assert normalize_optional_str("   ") is None

    def test_sentinel_returns_none(self):
        assert normalize_optional_str("none") is None
        assert normalize_optional_str("auto") is None
