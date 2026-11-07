"""Unit tests for AST static analyzer scripts/check_no_defaults_or_hardcode.py."""

from __future__ import annotations

import ast
from pathlib import Path

from scripts.check_no_defaults_or_hardcode import ZeroDefaultAndHardcodeVisitor


def _run_visitor(code: str) -> list[str]:
    tree = ast.parse(code)
    visitor = ZeroDefaultAndHardcodeVisitor(Path("test_file.py"))
    visitor.visit(tree)
    return visitor.violations


def test_no_default_func_param_triggers_violation() -> None:
    bad_code = (
        "def calculate_risk(price: float, risk_pct: float = 0.02):\n    return price * risk_pct\n"
    )
    violations = _run_visitor(bad_code)
    assert len(violations) == 1
    assert "[NO_DEFAULT_FUNC_PARAM]" in violations[0]


def test_no_default_kw_param_triggers_violation() -> None:
    bad_code = "def fetch_data(*, symbol: str, timeout: float = 5.0):\n    pass\n"
    violations = _run_visitor(bad_code)
    assert len(violations) == 1
    assert "[NO_DEFAULT_KW_PARAM]" in violations[0]


def test_no_default_class_field_in_pydantic_or_class() -> None:
    bad_code = (
        "class RiskConfig(BaseModel):\n"
        "    maker_fee_rate: float = 0.0002\n"
        "    taker_fee_rate: float\n"
    )
    violations = _run_visitor(bad_code)
    assert len(violations) == 1
    assert "[NO_DEFAULT_CLASS_FIELD]" in violations[0]
    assert "RiskConfig.maker_fee_rate" in violations[0]


def test_no_default_class_field_raw_assignment() -> None:
    bad_code = "class ConstantHolder:\n    TIMEOUT = 30\n"
    violations = _run_visitor(bad_code)
    assert len(violations) == 1
    assert "[NO_DEFAULT_CLASS_FIELD]" in violations[0]


def test_enum_subclasses_allow_member_assignments() -> None:
    clean_code = (
        "class TradeSide(str, Enum):\n    LONG = 'LONG'\n    SHORT = 'SHORT'\n    FLAT = 'FLAT'\n"
    )
    violations = _run_visitor(clean_code)
    assert len(violations) == 0


def test_no_dict_get_fallback_triggers_violation() -> None:
    bad_code = "val = payload.get('leverage', 1)\n"
    violations = _run_visitor(bad_code)
    assert len(violations) == 1
    assert "[NO_DICT_GET_FALLBACK]" in violations[0]


def test_dict_get_single_arg_allowed() -> None:
    clean_code = "val = payload.get('optional_meta')\n"
    violations = _run_visitor(clean_code)
    assert len(violations) == 0


def test_no_ternary_fallback_default_triggers_violation() -> None:
    bad_code = "leverage = raw_lev if raw_lev is not None else 1\n"
    violations = _run_visitor(bad_code)
    assert len(violations) == 1
    assert "[NO_TERNARY_FALLBACK_DEFAULT]" in violations[0]


def test_no_or_fallback_default_triggers_violation() -> None:
    bad_code = "vol = raw_vol or 0.015\n"
    violations = _run_visitor(bad_code)
    assert len(violations) == 1
    assert "[NO_OR_FALLBACK_DEFAULT]" in violations[0]


def test_fully_compliant_code_passes_without_violations() -> None:
    clean_code = (
        "class MarketConfig(BaseModel):\n"
        "    symbol: str\n"
        "    maker_fee_rate: float\n"
        "    rejection_reason: str | None\n"
        "\n"
        "def compute_sizing(config: MarketConfig, nav: float) -> float:\n"
        "    if nav <= 0.0:\n"
        "        raise ValueError('NAV must be positive')\n"
        "    return nav * config.maker_fee_rate\n"
    )
    violations = _run_visitor(clean_code)
    assert len(violations) == 0
