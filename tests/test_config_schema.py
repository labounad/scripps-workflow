"""Tests for ``scripps_workflow.config_schema``.

Pure-function tests for ConfigField + NodeSchema + apply_schema. The
framework is dependency-free (no rdkit / numpy / orca needed) so the
whole suite runs in any environment.
"""

from __future__ import annotations

import pytest

from scripps_workflow.config_schema import (
    REQUIRED,
    ConfigField,
    NodeSchema,
    apply_schema,
)


def _schema(*fields: ConfigField) -> NodeSchema:
    """Minimal NodeSchema builder for tests."""
    return NodeSchema(
        step_name="test",
        cli_entrypoint="wf-test",
        module_path="tests.fake",
        fields=tuple(fields),
    )


class TestTypeCoercion:
    @pytest.mark.parametrize(
        "type_str, raw_value, expected",
        [
            ("str", "  hello  ", "hello"),
            ("str", 42, "42"),
            ("int", "10", 10),
            ("int", 10, 10),
            ("int", "  -5  ", -5),
            ("float", "3.14", 3.14),
            ("float", 3, 3.0),
            ("bool", "true", True),
            ("bool", "false", False),
            ("bool", "yes", True),
            ("bool", "no", False),
            ("bool", 1, True),
            ("bool", 0, False),
            # NB: ("bool", "", False) is intentionally absent — apply_schema
            # treats "" as missing and uses the field default. The bool
            # coercer's empty-string → False fallback is reachable only by
            # calling the coercer directly. See
            # ``test_default_used_when_key_empty_string`` for the schema
            # layer's empty-string contract.
            ("csv", "a, b, c", ["a", "b", "c"]),
            ("csv", ["a", "b"], ["a", "b"]),
            ("csv", "single", ["single"]),
            ("json", '{"a": 1}', {"a": 1}),
            ("json", [1, 2, 3], [1, 2, 3]),
        ],
    )
    def test_coerce(self, type_str, raw_value, expected):
        schema = _schema(ConfigField(name="x", type=type_str, default=None))
        cfg = apply_schema({"x": raw_value}, schema)
        assert cfg["x"] == expected

    def test_invalid_bool_raises(self):
        schema = _schema(ConfigField(name="x", type="bool", default=None))
        with pytest.raises(ValueError, match="cannot coerce"):
            apply_schema({"x": "maybe"}, schema)


class TestDefaultsAndRequired:
    def test_default_used_when_key_absent(self):
        schema = _schema(ConfigField(name="x", type="int", default=42))
        assert apply_schema({}, schema)["x"] == 42

    def test_default_used_when_key_empty_string(self):
        schema = _schema(ConfigField(name="x", type="int", default=42))
        assert apply_schema({"x": ""}, schema)["x"] == 42

    def test_default_used_when_key_none(self):
        schema = _schema(ConfigField(name="x", type="int", default=42))
        assert apply_schema({"x": None}, schema)["x"] == 42

    def test_required_field_raises_when_missing(self):
        schema = _schema(ConfigField(name="x", type="str"))
        with pytest.raises(ValueError, match="required"):
            apply_schema({}, schema)


class TestAliases:
    def test_alias_lookup(self):
        schema = _schema(
            ConfigField(
                name="max_dE_kcal",
                type="float",
                default=0.5,
                aliases=("energy_window_kcal",),
            )
        )
        cfg = apply_schema({"energy_window_kcal": "1.0"}, schema)
        assert cfg["max_dE_kcal"] == 1.0
        # Canonical name takes precedence when both present.
        cfg = apply_schema(
            {"max_dE_kcal": "2.0", "energy_window_kcal": "1.0"}, schema
        )
        assert cfg["max_dE_kcal"] == 2.0


class TestChoicesAndRanges:
    def test_enum_accepts_valid_choice(self):
        schema = _schema(
            ConfigField(
                name="mode", type="enum", default="standard",
                choices=("standard", "fast", "thorough"),
            )
        )
        assert apply_schema({"mode": "fast"}, schema)["mode"] == "fast"

    def test_enum_rejects_invalid_choice(self):
        schema = _schema(
            ConfigField(
                name="mode", type="enum", default="standard",
                choices=("standard", "fast"),
            )
        )
        with pytest.raises(ValueError, match="must be one of"):
            apply_schema({"mode": "weird"}, schema)

    def test_min_value(self):
        schema = _schema(
            ConfigField(name="n", type="int", default=5, min_value=1)
        )
        with pytest.raises(ValueError, match="must be >= 1"):
            apply_schema({"n": "0"}, schema)
        assert apply_schema({"n": "1"}, schema)["n"] == 1

    def test_max_value(self):
        schema = _schema(
            ConfigField(name="n", type="int", default=5, max_value=10)
        )
        with pytest.raises(ValueError, match="must be <= 10"):
            apply_schema({"n": "11"}, schema)
        assert apply_schema({"n": "10"}, schema)["n"] == 10


class TestCustomCallbacks:
    def test_custom_coercer_accepts_valid(self):
        # Partner-list-style coercer with element whitelist.
        valid = {"H", "C", "N", "O", "F"}

        def parse_partners(raw):
            items = [s.strip() for s in str(raw).split(",") if s.strip()]
            for item in items:
                if item not in valid:
                    raise ValueError(f"unknown element {item!r}")
            return items

        schema = _schema(
            ConfigField(
                name="partners", type="csv", default=[],
                coercer=parse_partners,
            )
        )
        assert apply_schema({"partners": "F,O"}, schema)["partners"] == ["F", "O"]
        with pytest.raises(ValueError, match="partners: unknown element 'P'"):
            apply_schema({"partners": "F,P"}, schema)

    def test_custom_coercer_raises_through_apply(self):
        def coercer(raw):
            raise ValueError("bad")

        schema = _schema(
            ConfigField(name="x", type="str", default="", coercer=coercer)
        )
        with pytest.raises(ValueError, match="x: bad"):
            apply_schema({"x": "anything"}, schema)

    def test_validator_normalizes(self):
        def upper(value):
            return str(value).upper()

        schema = _schema(
            ConfigField(name="x", type="str", default="", validator=upper)
        )
        assert apply_schema({"x": "hello"}, schema)["x"] == "HELLO"

    def test_validator_can_raise(self):
        def must_be_basename(value):
            if "/" in value:
                raise ValueError("must be a basename, no slashes")
            return value

        schema = _schema(
            ConfigField(
                name="output_csv", type="str",
                default="x.csv", validator=must_be_basename,
            )
        )
        with pytest.raises(ValueError, match="must be a basename"):
            apply_schema({"output_csv": "a/b.csv"}, schema)


class TestNodeSchemaIntrospection:
    def test_field_by_name(self):
        f1 = ConfigField(name="a", type="str", default="x")
        f2 = ConfigField(
            name="b", type="int", default=1, aliases=("b_alias",)
        )
        schema = _schema(f1, f2)
        assert schema.field_by_name("a") is f1
        assert schema.field_by_name("b") is f2
        assert schema.field_by_name("b_alias") is f2
        assert schema.field_by_name("nope") is None

    def test_is_required(self):
        required = ConfigField(name="x", type="str")
        defaulted = ConfigField(name="y", type="str", default="")
        assert required.is_required() is True
        assert defaulted.is_required() is False
