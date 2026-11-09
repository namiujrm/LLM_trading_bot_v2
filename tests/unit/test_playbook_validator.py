"""Unit tests for PlaybookValidator enhancements.

Tests SHORT_COVERING quadrant acceptance, classify_and_validate tactical waterfall,
and GENERAL_ALPHA_MOMENTUM fallback.
"""

from __future__ import annotations

import pytest

from trading_bot.domain.entities.common.enums import OIPriceQuadrant, PlaybookType, TradeSide
from trading_bot.domain.services.execution.playbook_validator import (
    PlaybookConfig,
    PlaybookPreconditionsInput,
    PlaybookValidator,
)


@pytest.fixture
def relaxed_playbook_config() -> PlaybookConfig:
    return PlaybookConfig(
        trend_hurst_min=0.50,  # Relaxed to 0.50
        trend_vpin_percentile_max=88.0,
        trend_min_calibrated_prob=0.52,  # Relaxed to 0.52
        reversal_hurst_max=0.49,
        reversal_liq_z_min=1.8,
        reversal_funding_z_abs_min=2.0,
        reversal_vpin_slope_max=0.0,
        lead_lag_btc_ret_z_min=1.5,
        lead_lag_btc_beta_min=0.50,
        lead_lag_catchup_gap_z_min=1.0,
        altcoin_symbols=["ETHUSDT", "SOLUSDT", "BNBUSDT"],
    )


def test_trend_continuation_accepts_short_covering(
    relaxed_playbook_config: PlaybookConfig,
) -> None:
    validator = PlaybookValidator(relaxed_playbook_config)
    inp = PlaybookPreconditionsInput(
        symbol="BTCUSDT",
        hurst_exp=0.55,
        oi_quadrant=OIPriceQuadrant.SHORT_COVERING,  # Short covering quadrant
        vpin_percentile=70.0,
        p_long_cal=0.60,
        p_short_cal=0.40,
        liq_long_z=0.0,
        liq_short_z=0.0,
        funding_z=0.0,
        vpin_slope_3bars=0.0,
        btc_15m_return_z=0.0,
        btc_lead_beta=1.0,
        lead_lag_catchup_gap_z=0.0,
        is_ood_warning=False,
        pre_event_blackout=False,
    )
    result = validator.verify_playbook_preconditions(PlaybookType.TREND_CONTINUATION_BREAKOUT, inp)
    assert result.is_valid is True
    assert result.playbook == PlaybookType.TREND_CONTINUATION_BREAKOUT
    assert result.allowed_side == TradeSide.LONG
    assert result.reason == "TREND_CONTINUATION_PRECONDITIONS_SATISFIED"


def test_classify_and_validate_ood_forces_no_trade(
    relaxed_playbook_config: PlaybookConfig,
) -> None:
    validator = PlaybookValidator(relaxed_playbook_config)
    inp = PlaybookPreconditionsInput(
        symbol="BTCUSDT",
        hurst_exp=0.55,
        oi_quadrant=OIPriceQuadrant.LONG_BUILDUP,
        vpin_percentile=70.0,
        p_long_cal=0.60,
        p_short_cal=0.40,
        liq_long_z=0.0,
        liq_short_z=0.0,
        funding_z=0.0,
        vpin_slope_3bars=0.0,
        btc_15m_return_z=0.0,
        btc_lead_beta=1.0,
        lead_lag_catchup_gap_z=0.0,
        is_ood_warning=True,  # OOD Active
        pre_event_blackout=False,
    )
    res = validator.classify_and_validate(inp)
    assert res.is_valid is False
    assert res.playbook == PlaybookType.NO_TRADE_REGIME
    assert res.allowed_side == TradeSide.FLAT
    assert res.reason == "OUT_OF_DISTRIBUTION_WARNING_ACTIVE"


def test_classify_and_validate_blackout_forces_no_trade(
    relaxed_playbook_config: PlaybookConfig,
) -> None:
    validator = PlaybookValidator(relaxed_playbook_config)
    inp = PlaybookPreconditionsInput(
        symbol="BTCUSDT",
        hurst_exp=0.55,
        oi_quadrant=OIPriceQuadrant.LONG_BUILDUP,
        vpin_percentile=70.0,
        p_long_cal=0.60,
        p_short_cal=0.40,
        liq_long_z=0.0,
        liq_short_z=0.0,
        funding_z=0.0,
        vpin_slope_3bars=0.0,
        btc_15m_return_z=0.0,
        btc_lead_beta=1.0,
        lead_lag_catchup_gap_z=0.0,
        is_ood_warning=False,
        pre_event_blackout=True,  # Macro blackout
    )
    res = validator.classify_and_validate(inp)
    assert res.is_valid is False
    assert res.playbook == PlaybookType.NO_TRADE_REGIME
    assert res.allowed_side == TradeSide.FLAT
    assert res.reason == "MACRO_EVENT_BLACKOUT_WINDOW_ACTIVE"


def test_classify_and_validate_selects_trend(
    relaxed_playbook_config: PlaybookConfig,
) -> None:
    validator = PlaybookValidator(relaxed_playbook_config)
    inp = PlaybookPreconditionsInput(
        symbol="BTCUSDT",
        hurst_exp=0.55,
        oi_quadrant=OIPriceQuadrant.LONG_BUILDUP,
        vpin_percentile=70.0,
        p_long_cal=0.58,
        p_short_cal=0.42,
        liq_long_z=0.0,
        liq_short_z=0.0,
        funding_z=0.0,
        vpin_slope_3bars=0.0,
        btc_15m_return_z=0.0,
        btc_lead_beta=1.0,
        lead_lag_catchup_gap_z=0.0,
        is_ood_warning=False,
        pre_event_blackout=False,
    )
    res = validator.classify_and_validate(inp)
    assert res.is_valid is True
    assert res.playbook == PlaybookType.TREND_CONTINUATION_BREAKOUT
    assert res.allowed_side == TradeSide.LONG


def test_classify_and_validate_selects_reversal(
    relaxed_playbook_config: PlaybookConfig,
) -> None:
    validator = PlaybookValidator(relaxed_playbook_config)
    # Fails trend continuation (hurst < 0.50), but passes liquidation exhaustion
    inp = PlaybookPreconditionsInput(
        symbol="BTCUSDT",
        hurst_exp=0.45,
        oi_quadrant=OIPriceQuadrant.LONG_LIQUIDATION,
        vpin_percentile=70.0,
        p_long_cal=0.48,
        p_short_cal=0.52,
        liq_long_z=2.5,  # Long liquidation spike
        liq_short_z=0.0,
        funding_z=-2.2,
        vpin_slope_3bars=-0.05,
        btc_15m_return_z=0.0,
        btc_lead_beta=1.0,
        lead_lag_catchup_gap_z=0.0,
        is_ood_warning=False,
        pre_event_blackout=False,
    )
    res = validator.classify_and_validate(inp)
    assert res.is_valid is True
    assert res.playbook == PlaybookType.LIQUIDATION_EXHAUSTION_REVERSAL
    assert res.allowed_side == TradeSide.LONG


def test_classify_and_validate_selects_lead_lag(
    relaxed_playbook_config: PlaybookConfig,
) -> None:
    validator = PlaybookValidator(relaxed_playbook_config)
    # Fails trend (Hurst 0.49) and fails reversal (no liq spike), but passes lead-lag
    inp = PlaybookPreconditionsInput(
        symbol="ETHUSDT",
        hurst_exp=0.49,
        oi_quadrant=OIPriceQuadrant.NEUTRAL,
        vpin_percentile=70.0,
        p_long_cal=0.50,
        p_short_cal=0.50,
        liq_long_z=0.0,
        liq_short_z=0.0,
        funding_z=0.0,
        vpin_slope_3bars=0.0,
        btc_15m_return_z=2.0,  # BTC breakout
        btc_lead_beta=0.75,
        lead_lag_catchup_gap_z=1.2,
        is_ood_warning=False,
        pre_event_blackout=False,
    )
    res = validator.classify_and_validate(inp)
    assert res.is_valid is True
    assert res.playbook == PlaybookType.CROSS_ASSET_LEAD_LAG_CATCHUP
    assert res.allowed_side == TradeSide.LONG


def test_classify_and_validate_general_alpha_momentum_long(
    relaxed_playbook_config: PlaybookConfig,
) -> None:
    validator = PlaybookValidator(relaxed_playbook_config)
    # Doesn't match Trend (neutral OI), Reversal (no liq spike), or Lead-Lag (not an altcoin or zero btc_z)
    # But has p_long_cal = 0.56 >= 0.52 -> GENERAL_ALPHA_MOMENTUM
    inp = PlaybookPreconditionsInput(
        symbol="BTCUSDT",
        hurst_exp=0.49,
        oi_quadrant=OIPriceQuadrant.NEUTRAL,
        vpin_percentile=70.0,
        p_long_cal=0.56,
        p_short_cal=0.44,
        liq_long_z=0.0,
        liq_short_z=0.0,
        funding_z=0.0,
        vpin_slope_3bars=0.0,
        btc_15m_return_z=0.0,
        btc_lead_beta=1.0,
        lead_lag_catchup_gap_z=0.0,
        is_ood_warning=False,
        pre_event_blackout=False,
    )
    res = validator.classify_and_validate(inp)
    assert res.is_valid is True
    assert res.playbook == PlaybookType.GENERAL_ALPHA_MOMENTUM
    assert res.allowed_side == TradeSide.LONG
    assert res.reason == "GENERAL_ALPHA_MOMENTUM_CLEARED"


def test_classify_and_validate_general_alpha_momentum_short(
    relaxed_playbook_config: PlaybookConfig,
) -> None:
    validator = PlaybookValidator(relaxed_playbook_config)
    inp = PlaybookPreconditionsInput(
        symbol="BTCUSDT",
        hurst_exp=0.49,
        oi_quadrant=OIPriceQuadrant.NEUTRAL,
        vpin_percentile=70.0,
        p_long_cal=0.40,
        p_short_cal=0.60,  # 0.60 >= 0.52
        liq_long_z=0.0,
        liq_short_z=0.0,
        funding_z=0.0,
        vpin_slope_3bars=0.0,
        btc_15m_return_z=0.0,
        btc_lead_beta=1.0,
        lead_lag_catchup_gap_z=0.0,
        is_ood_warning=False,
        pre_event_blackout=False,
    )
    res = validator.classify_and_validate(inp)
    assert res.is_valid is True
    assert res.playbook == PlaybookType.GENERAL_ALPHA_MOMENTUM
    assert res.allowed_side == TradeSide.SHORT
    assert res.reason == "GENERAL_ALPHA_MOMENTUM_CLEARED"


def test_classify_and_validate_falls_back_to_no_trade(
    relaxed_playbook_config: PlaybookConfig,
) -> None:
    validator = PlaybookValidator(relaxed_playbook_config)
    # Neither matches templates NOR has sufficient conviction (max prob 0.51 < 0.52)
    inp = PlaybookPreconditionsInput(
        symbol="BTCUSDT",
        hurst_exp=0.49,
        oi_quadrant=OIPriceQuadrant.NEUTRAL,
        vpin_percentile=70.0,
        p_long_cal=0.51,
        p_short_cal=0.49,
        liq_long_z=0.0,
        liq_short_z=0.0,
        funding_z=0.0,
        vpin_slope_3bars=0.0,
        btc_15m_return_z=0.0,
        btc_lead_beta=1.0,
        lead_lag_catchup_gap_z=0.0,
        is_ood_warning=False,
        pre_event_blackout=False,
    )
    res = validator.classify_and_validate(inp)
    assert res.is_valid is False
    assert res.playbook == PlaybookType.NO_TRADE_REGIME
    assert res.allowed_side == TradeSide.FLAT
    assert res.reason == "NO_PLAYBOOK_PRECONDITIONS_MET"


def test_verify_playbook_preconditions_general_alpha(
    relaxed_playbook_config: PlaybookConfig,
) -> None:
    validator = PlaybookValidator(relaxed_playbook_config)
    inp = PlaybookPreconditionsInput(
        symbol="BTCUSDT",
        hurst_exp=0.49,
        oi_quadrant=OIPriceQuadrant.NEUTRAL,
        vpin_percentile=70.0,
        p_long_cal=0.55,
        p_short_cal=0.45,
        liq_long_z=0.0,
        liq_short_z=0.0,
        funding_z=0.0,
        vpin_slope_3bars=0.0,
        btc_15m_return_z=0.0,
        btc_lead_beta=1.0,
        lead_lag_catchup_gap_z=0.0,
        is_ood_warning=False,
        pre_event_blackout=False,
    )
    res = validator.verify_playbook_preconditions(PlaybookType.GENERAL_ALPHA_MOMENTUM, inp)
    assert res.is_valid is True
    assert res.playbook == PlaybookType.GENERAL_ALPHA_MOMENTUM
    assert res.allowed_side == TradeSide.LONG
