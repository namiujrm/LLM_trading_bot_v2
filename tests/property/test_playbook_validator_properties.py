"""Hypothesis property-based tests for PlaybookValidator."""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from trading_bot.domain.entities.common.enums import OIPriceQuadrant, PlaybookType, TradeSide
from trading_bot.domain.services.execution.playbook_validator import (
    PlaybookConfig,
    PlaybookPreconditionsInput,
    PlaybookValidator,
)

CONFIG = PlaybookConfig(
    trend_hurst_min=0.54,
    trend_vpin_percentile_max=88.0,
    trend_min_calibrated_prob=0.57,
    reversal_hurst_max=0.49,
    reversal_liq_z_min=2.0,
    reversal_funding_z_abs_min=2.2,
    reversal_vpin_slope_max=0.0,
    lead_lag_btc_ret_z_min=1.8,
    lead_lag_btc_beta_min=0.55,
    lead_lag_catchup_gap_z_min=1.5,
    altcoin_symbols=["ETHUSDT", "SOLUSDT", "BNBUSDT"],
)


@settings(max_examples=100)
@given(
    hurst=st.floats(min_value=0.20, max_value=0.80),
    oi_quad=st.sampled_from(list(OIPriceQuadrant)),
    vpin_pct=st.floats(min_value=1.0, max_value=99.0),
    p_long=st.floats(min_value=0.20, max_value=0.80),
    p_short=st.floats(min_value=0.20, max_value=0.80),
    liq_long_z=st.floats(min_value=-3.0, max_value=5.0),
    liq_short_z=st.floats(min_value=-3.0, max_value=5.0),
    funding_z=st.floats(min_value=-4.0, max_value=4.0),
    vpin_slope=st.floats(min_value=-1.0, max_value=1.0),
    btc_ret_z=st.floats(min_value=-4.0, max_value=4.0),
    btc_beta=st.floats(min_value=0.0, max_value=2.0),
    catchup_gap_z=st.floats(min_value=-4.0, max_value=4.0),
    is_ood=st.booleans(),
    blackout=st.booleans(),
    symbol=st.sampled_from(["BTCUSDT", "ETHUSDT", "SOLUSDT"]),
    playbook=st.sampled_from(list(PlaybookType)),
)
def test_playbook_validator_invariants(
    hurst: float,
    oi_quad: OIPriceQuadrant,
    vpin_pct: float,
    p_long: float,
    p_short: float,
    liq_long_z: float,
    liq_short_z: float,
    funding_z: float,
    vpin_slope: float,
    btc_ret_z: float,
    btc_beta: float,
    catchup_gap_z: float,
    is_ood: bool,
    blackout: bool,
    symbol: str,
    playbook: PlaybookType,
) -> None:
    validator = PlaybookValidator(CONFIG)
    inp = PlaybookPreconditionsInput(
        symbol=symbol,
        hurst_exp=hurst,
        oi_quadrant=oi_quad,
        vpin_percentile=vpin_pct,
        p_long_cal=p_long,
        p_short_cal=p_short,
        liq_long_z=liq_long_z,
        liq_short_z=liq_short_z,
        funding_z=funding_z,
        vpin_slope_3bars=vpin_slope,
        btc_15m_return_z=btc_ret_z,
        btc_lead_beta=btc_beta,
        lead_lag_catchup_gap_z=catchup_gap_z,
        is_ood_warning=is_ood,
        pre_event_blackout=blackout,
    )

    res = validator.verify_playbook_preconditions(playbook, inp)

    # Invariant 1: OOD warning or Blackout MUST force NO_TRADE_REGIME and FLAT
    if is_ood or blackout:
        assert res.is_valid is False
        assert res.playbook == PlaybookType.NO_TRADE_REGIME
        assert res.allowed_side == TradeSide.FLAT

    # Invariant 2: If Trend Continuation is valid, hurst and vpin must satisfy bounds
    if res.is_valid and res.playbook == PlaybookType.TREND_CONTINUATION_BREAKOUT:
        assert hurst >= CONFIG.trend_hurst_min
        assert vpin_pct <= CONFIG.trend_vpin_percentile_max
        assert max(p_long, p_short) >= CONFIG.trend_min_calibrated_prob
        assert oi_quad in (
            OIPriceQuadrant.LONG_BUILDUP,
            OIPriceQuadrant.SHORT_BUILDUP,
            OIPriceQuadrant.SHORT_COVERING,
        )
