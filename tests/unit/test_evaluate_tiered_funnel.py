"""Unit tests for TieredFunnelEvaluator application use case.

Verifies:
- Stage A rejections (spread, Yang-Zhang vol, macro blackout, account freeze)
- Stage B rejections (OOD warning, Venn-Abers uncertainty, fee-to-edge hurdle)
- Stage C classifications (trend continuation, liquidation reversal, lead-lag catchup, general alpha momentum)
- End-to-end full pass verdicts
- Zero Defaults compliance (Invariant #9)
"""

from __future__ import annotations

import pytest

from trading_bot.application.use_cases.evaluate_tiered_funnel import (
    FunnelStage,
    TieredFunnelEvaluator,
    TieredFunnelInput,
    TieredFunnelVerdict,
)
from trading_bot.domain.entities.common.enums import (
    OIPriceQuadrant,
    PlaybookType,
    TradeSide,
)
from trading_bot.domain.services.execution.fee_to_edge_gate import (
    FeeToEdgeConfig,
    FeeToEdgeGatingService,
)
from trading_bot.domain.services.execution.playbook_validator import (
    PlaybookConfig,
    PlaybookValidator,
)


@pytest.fixture
def fee_to_edge_service() -> FeeToEdgeGatingService:
    config = FeeToEdgeConfig(
        multiplier=1.0,
        roundtrip_cost_rate=0.0010,  # 10 bps
        min_calibrated_prob=0.52,
    )
    return FeeToEdgeGatingService(config)


@pytest.fixture
def playbook_validator() -> PlaybookValidator:
    config = PlaybookConfig(
        trend_hurst_min=0.50,
        trend_vpin_percentile_max=88.0,
        trend_min_calibrated_prob=0.52,
        reversal_hurst_max=0.49,
        reversal_liq_z_min=1.8,
        reversal_funding_z_abs_min=2.0,
        reversal_vpin_slope_max=0.0,
        lead_lag_btc_ret_z_min=1.5,
        lead_lag_btc_beta_min=0.50,
        lead_lag_catchup_gap_z_min=1.0,
        altcoin_symbols=["ETHUSDT", "SOLUSDT", "BNBUSDT"],
    )
    return PlaybookValidator(config)


@pytest.fixture
def funnel_evaluator(
    fee_to_edge_service: FeeToEdgeGatingService,
    playbook_validator: PlaybookValidator,
) -> TieredFunnelEvaluator:
    return TieredFunnelEvaluator(
        fee_to_edge_service=fee_to_edge_service,
        playbook_validator=playbook_validator,
        max_spread_bps=5.0,
        min_yz_vol=0.002,
        max_uncertainty_width=0.15,
    )


def make_valid_trend_input() -> TieredFunnelInput:
    """Helper creating a baseline TieredFunnelInput without default parameters."""
    return TieredFunnelInput(
        symbol="BTCUSDT",
        spread_bps=2.0,
        yang_zhang_vol=0.015,
        garman_klass_vol=0.015,
        is_blackout=False,
        is_account_frozen=False,
        p_long_cal=0.62,
        p_short_cal=0.38,
        uncertainty_width=0.08,
        is_ood=False,
        hurst_exp=0.58,
        oi_quadrant=OIPriceQuadrant.LONG_BUILDUP,
        vpin_percentile=60.0,
        liq_long_z=0.2,
        liq_short_z=0.1,
        funding_z=0.5,
        vpin_slope_3bars=-0.05,
        btc_15m_return_z=0.0,
        btc_lead_beta=1.0,
        lead_lag_catchup_gap_z=0.0,
    )


# ==============================================================================
# Stage A Rejections
# ==============================================================================


def test_funnel_rejects_on_account_frozen(
    funnel_evaluator: TieredFunnelEvaluator,
) -> None:
    base = make_valid_trend_input()
    inp = TieredFunnelInput(
        symbol=base.symbol,
        spread_bps=base.spread_bps,
        yang_zhang_vol=base.yang_zhang_vol,
        garman_klass_vol=base.garman_klass_vol,
        is_blackout=base.is_blackout,
        is_account_frozen=True,
        p_long_cal=base.p_long_cal,
        p_short_cal=base.p_short_cal,
        uncertainty_width=base.uncertainty_width,
        is_ood=base.is_ood,
        hurst_exp=base.hurst_exp,
        oi_quadrant=base.oi_quadrant,
        vpin_percentile=base.vpin_percentile,
        liq_long_z=base.liq_long_z,
        liq_short_z=base.liq_short_z,
        funding_z=base.funding_z,
        vpin_slope_3bars=base.vpin_slope_3bars,
        btc_15m_return_z=base.btc_15m_return_z,
        btc_lead_beta=base.btc_lead_beta,
        lead_lag_catchup_gap_z=base.lead_lag_catchup_gap_z,
    )
    verdict = funnel_evaluator.evaluate(inp)
    assert isinstance(verdict, TieredFunnelVerdict)
    assert verdict.passed is False
    assert verdict.stage == FunnelStage.STAGE_A_RULES
    assert verdict.playbook == PlaybookType.NO_TRADE_REGIME
    assert verdict.allowed_side == TradeSide.FLAT
    assert verdict.rejection_reason == "ACCOUNT_FROZEN"


def test_funnel_rejects_on_macro_blackout(
    funnel_evaluator: TieredFunnelEvaluator,
) -> None:
    base = make_valid_trend_input()
    inp = TieredFunnelInput(
        symbol=base.symbol,
        spread_bps=base.spread_bps,
        yang_zhang_vol=base.yang_zhang_vol,
        garman_klass_vol=base.garman_klass_vol,
        is_blackout=True,
        is_account_frozen=False,
        p_long_cal=base.p_long_cal,
        p_short_cal=base.p_short_cal,
        uncertainty_width=base.uncertainty_width,
        is_ood=base.is_ood,
        hurst_exp=base.hurst_exp,
        oi_quadrant=base.oi_quadrant,
        vpin_percentile=base.vpin_percentile,
        liq_long_z=base.liq_long_z,
        liq_short_z=base.liq_short_z,
        funding_z=base.funding_z,
        vpin_slope_3bars=base.vpin_slope_3bars,
        btc_15m_return_z=base.btc_15m_return_z,
        btc_lead_beta=base.btc_lead_beta,
        lead_lag_catchup_gap_z=base.lead_lag_catchup_gap_z,
    )
    verdict = funnel_evaluator.evaluate(inp)
    assert verdict.passed is False
    assert verdict.stage == FunnelStage.STAGE_A_RULES
    assert verdict.rejection_reason == "MACRO_EVENT_BLACKOUT"


def test_funnel_rejects_on_wide_spread(
    funnel_evaluator: TieredFunnelEvaluator,
) -> None:
    base = make_valid_trend_input()
    inp = TieredFunnelInput(
        symbol=base.symbol,
        spread_bps=8.5,  # Exceeds max_spread_bps=5.0
        yang_zhang_vol=base.yang_zhang_vol,
        garman_klass_vol=base.garman_klass_vol,
        is_blackout=False,
        is_account_frozen=False,
        p_long_cal=base.p_long_cal,
        p_short_cal=base.p_short_cal,
        uncertainty_width=base.uncertainty_width,
        is_ood=base.is_ood,
        hurst_exp=base.hurst_exp,
        oi_quadrant=base.oi_quadrant,
        vpin_percentile=base.vpin_percentile,
        liq_long_z=base.liq_long_z,
        liq_short_z=base.liq_short_z,
        funding_z=base.funding_z,
        vpin_slope_3bars=base.vpin_slope_3bars,
        btc_15m_return_z=base.btc_15m_return_z,
        btc_lead_beta=base.btc_lead_beta,
        lead_lag_catchup_gap_z=base.lead_lag_catchup_gap_z,
    )
    verdict = funnel_evaluator.evaluate(inp)
    assert verdict.passed is False
    assert verdict.stage == FunnelStage.STAGE_A_RULES
    assert verdict.rejection_reason is not None
    assert "SPREAD_TOO_WIDE" in verdict.rejection_reason


def test_funnel_rejects_on_low_yang_zhang_vol(
    funnel_evaluator: TieredFunnelEvaluator,
) -> None:
    base = make_valid_trend_input()
    inp = TieredFunnelInput(
        symbol=base.symbol,
        spread_bps=base.spread_bps,
        yang_zhang_vol=0.0008,  # Below min_yz_vol=0.002
        garman_klass_vol=base.garman_klass_vol,
        is_blackout=False,
        is_account_frozen=False,
        p_long_cal=base.p_long_cal,
        p_short_cal=base.p_short_cal,
        uncertainty_width=base.uncertainty_width,
        is_ood=base.is_ood,
        hurst_exp=base.hurst_exp,
        oi_quadrant=base.oi_quadrant,
        vpin_percentile=base.vpin_percentile,
        liq_long_z=base.liq_long_z,
        liq_short_z=base.liq_short_z,
        funding_z=base.funding_z,
        vpin_slope_3bars=base.vpin_slope_3bars,
        btc_15m_return_z=base.btc_15m_return_z,
        btc_lead_beta=base.btc_lead_beta,
        lead_lag_catchup_gap_z=base.lead_lag_catchup_gap_z,
    )
    verdict = funnel_evaluator.evaluate(inp)
    assert verdict.passed is False
    assert verdict.stage == FunnelStage.STAGE_A_RULES
    assert verdict.rejection_reason is not None
    assert "YZ_VOL_BELOW_MIN" in verdict.rejection_reason


# ==============================================================================
# Stage B Rejections
# ==============================================================================


def test_funnel_rejects_on_ood_warning(
    funnel_evaluator: TieredFunnelEvaluator,
) -> None:
    base = make_valid_trend_input()
    inp = TieredFunnelInput(
        symbol=base.symbol,
        spread_bps=base.spread_bps,
        yang_zhang_vol=base.yang_zhang_vol,
        garman_klass_vol=base.garman_klass_vol,
        is_blackout=False,
        is_account_frozen=False,
        p_long_cal=base.p_long_cal,
        p_short_cal=base.p_short_cal,
        uncertainty_width=base.uncertainty_width,
        is_ood=True,
        hurst_exp=base.hurst_exp,
        oi_quadrant=base.oi_quadrant,
        vpin_percentile=base.vpin_percentile,
        liq_long_z=base.liq_long_z,
        liq_short_z=base.liq_short_z,
        funding_z=base.funding_z,
        vpin_slope_3bars=base.vpin_slope_3bars,
        btc_15m_return_z=base.btc_15m_return_z,
        btc_lead_beta=base.btc_lead_beta,
        lead_lag_catchup_gap_z=base.lead_lag_catchup_gap_z,
    )
    verdict = funnel_evaluator.evaluate(inp)
    assert verdict.passed is False
    assert verdict.stage == FunnelStage.STAGE_B_FEE_EDGE
    assert verdict.rejection_reason == "OUT_OF_DISTRIBUTION_WARNING"


def test_funnel_rejects_on_wide_uncertainty(
    funnel_evaluator: TieredFunnelEvaluator,
) -> None:
    base = make_valid_trend_input()
    inp = TieredFunnelInput(
        symbol=base.symbol,
        spread_bps=base.spread_bps,
        yang_zhang_vol=base.yang_zhang_vol,
        garman_klass_vol=base.garman_klass_vol,
        is_blackout=False,
        is_account_frozen=False,
        p_long_cal=base.p_long_cal,
        p_short_cal=base.p_short_cal,
        uncertainty_width=0.22,  # Exceeds max_uncertainty_width=0.15
        is_ood=False,
        hurst_exp=base.hurst_exp,
        oi_quadrant=base.oi_quadrant,
        vpin_percentile=base.vpin_percentile,
        liq_long_z=base.liq_long_z,
        liq_short_z=base.liq_short_z,
        funding_z=base.funding_z,
        vpin_slope_3bars=base.vpin_slope_3bars,
        btc_15m_return_z=base.btc_15m_return_z,
        btc_lead_beta=base.btc_lead_beta,
        lead_lag_catchup_gap_z=base.lead_lag_catchup_gap_z,
    )
    verdict = funnel_evaluator.evaluate(inp)
    assert verdict.passed is False
    assert verdict.stage == FunnelStage.STAGE_B_FEE_EDGE
    assert verdict.rejection_reason is not None
    assert "UNCERTAINTY_TOO_WIDE" in verdict.rejection_reason


def test_funnel_rejects_on_fee_to_edge_conviction_low(
    funnel_evaluator: TieredFunnelEvaluator,
) -> None:
    base = make_valid_trend_input()
    inp = TieredFunnelInput(
        symbol=base.symbol,
        spread_bps=base.spread_bps,
        yang_zhang_vol=base.yang_zhang_vol,
        garman_klass_vol=base.garman_klass_vol,
        is_blackout=False,
        is_account_frozen=False,
        p_long_cal=0.51,  # Both < 0.52 threshold
        p_short_cal=0.49,
        uncertainty_width=base.uncertainty_width,
        is_ood=False,
        hurst_exp=base.hurst_exp,
        oi_quadrant=base.oi_quadrant,
        vpin_percentile=base.vpin_percentile,
        liq_long_z=base.liq_long_z,
        liq_short_z=base.liq_short_z,
        funding_z=base.funding_z,
        vpin_slope_3bars=base.vpin_slope_3bars,
        btc_15m_return_z=base.btc_15m_return_z,
        btc_lead_beta=base.btc_lead_beta,
        lead_lag_catchup_gap_z=base.lead_lag_catchup_gap_z,
    )
    verdict = funnel_evaluator.evaluate(inp)
    assert verdict.passed is False
    assert verdict.stage == FunnelStage.STAGE_B_FEE_EDGE
    assert verdict.rejection_reason == "CONVICTION_BELOW_MIN"


def test_funnel_rejects_on_fee_to_edge_below_hurdle(
    funnel_evaluator: TieredFunnelEvaluator,
) -> None:
    base = make_valid_trend_input()
    inp = TieredFunnelInput(
        symbol=base.symbol,
        spread_bps=base.spread_bps,
        yang_zhang_vol=base.yang_zhang_vol,
        garman_klass_vol=0.0005,  # Very low vol: expected edge below 10 bps hurdle
        is_blackout=False,
        is_account_frozen=False,
        p_long_cal=0.53,  # Passes 0.52 conviction but edge is thin
        p_short_cal=0.47,
        uncertainty_width=base.uncertainty_width,
        is_ood=False,
        hurst_exp=base.hurst_exp,
        oi_quadrant=base.oi_quadrant,
        vpin_percentile=base.vpin_percentile,
        liq_long_z=base.liq_long_z,
        liq_short_z=base.liq_short_z,
        funding_z=base.funding_z,
        vpin_slope_3bars=base.vpin_slope_3bars,
        btc_15m_return_z=base.btc_15m_return_z,
        btc_lead_beta=base.btc_lead_beta,
        lead_lag_catchup_gap_z=base.lead_lag_catchup_gap_z,
    )
    verdict = funnel_evaluator.evaluate(inp)
    assert verdict.passed is False
    assert verdict.stage == FunnelStage.STAGE_B_FEE_EDGE
    assert verdict.rejection_reason == "EXPECTED_EDGE_BELOW_FEE_HURDLE"


# ==============================================================================
# Stage C Classifications and Validations
# ==============================================================================


def test_funnel_classifies_trend_continuation_long(
    funnel_evaluator: TieredFunnelEvaluator,
) -> None:
    inp = make_valid_trend_input()
    verdict = funnel_evaluator.evaluate(inp)
    assert verdict.passed is True
    assert verdict.stage == FunnelStage.PASSED
    assert verdict.playbook == PlaybookType.TREND_CONTINUATION_BREAKOUT
    assert verdict.allowed_side == TradeSide.LONG
    assert verdict.rejection_reason is None
    assert verdict.expected_edge_bps > 10.0


def test_funnel_classifies_trend_continuation_short(
    funnel_evaluator: TieredFunnelEvaluator,
) -> None:
    base = make_valid_trend_input()
    inp = TieredFunnelInput(
        symbol=base.symbol,
        spread_bps=base.spread_bps,
        yang_zhang_vol=base.yang_zhang_vol,
        garman_klass_vol=base.garman_klass_vol,
        is_blackout=False,
        is_account_frozen=False,
        p_long_cal=0.35,
        p_short_cal=0.65,
        uncertainty_width=base.uncertainty_width,
        is_ood=False,
        hurst_exp=0.56,
        oi_quadrant=OIPriceQuadrant.SHORT_BUILDUP,
        vpin_percentile=55.0,
        liq_long_z=0.0,
        liq_short_z=0.0,
        funding_z=0.0,
        vpin_slope_3bars=0.0,
        btc_15m_return_z=0.0,
        btc_lead_beta=1.0,
        lead_lag_catchup_gap_z=0.0,
    )
    verdict = funnel_evaluator.evaluate(inp)
    assert verdict.passed is True
    assert verdict.stage == FunnelStage.PASSED
    assert verdict.playbook == PlaybookType.TREND_CONTINUATION_BREAKOUT
    assert verdict.allowed_side == TradeSide.SHORT
    assert verdict.rejection_reason is None


def test_funnel_classifies_liquidation_exhaustion_reversal(
    funnel_evaluator: TieredFunnelEvaluator,
) -> None:
    base = make_valid_trend_input()
    inp = TieredFunnelInput(
        symbol=base.symbol,
        spread_bps=base.spread_bps,
        yang_zhang_vol=base.yang_zhang_vol,
        garman_klass_vol=base.garman_klass_vol,
        is_blackout=False,
        is_account_frozen=False,
        p_long_cal=0.60,
        p_short_cal=0.40,
        uncertainty_width=base.uncertainty_width,
        is_ood=False,
        hurst_exp=0.45,  # Mean reverting (< 0.49)
        oi_quadrant=OIPriceQuadrant.NEUTRAL,
        vpin_percentile=60.0,
        liq_long_z=2.5,  # Liq spike >= 1.8
        liq_short_z=0.1,
        funding_z=-2.2,
        vpin_slope_3bars=-0.1,  # Non-accelerating <= 0.0
        btc_15m_return_z=0.0,
        btc_lead_beta=1.0,
        lead_lag_catchup_gap_z=0.0,
    )
    verdict = funnel_evaluator.evaluate(inp)
    assert verdict.passed is True
    assert verdict.stage == FunnelStage.PASSED
    assert verdict.playbook == PlaybookType.LIQUIDATION_EXHAUSTION_REVERSAL
    assert verdict.allowed_side == TradeSide.LONG
    assert verdict.rejection_reason is None


def test_funnel_classifies_cross_asset_lead_lag_catchup(
    funnel_evaluator: TieredFunnelEvaluator,
) -> None:
    base = make_valid_trend_input()
    inp = TieredFunnelInput(
        symbol="ETHUSDT",  # Altcoin
        spread_bps=base.spread_bps,
        yang_zhang_vol=base.yang_zhang_vol,
        garman_klass_vol=base.garman_klass_vol,
        is_blackout=False,
        is_account_frozen=False,
        p_long_cal=0.56,
        p_short_cal=0.44,
        uncertainty_width=base.uncertainty_width,
        is_ood=False,
        hurst_exp=0.45,  # Not trend continuation
        oi_quadrant=OIPriceQuadrant.NEUTRAL,
        vpin_percentile=60.0,
        liq_long_z=0.1,  # Not liquidation reversal
        liq_short_z=0.1,
        funding_z=0.0,
        vpin_slope_3bars=0.0,
        btc_15m_return_z=2.2,  # BTC return z >= 1.5
        btc_lead_beta=0.90,  # Beta >= 0.50
        lead_lag_catchup_gap_z=1.8,  # Gap >= 1.0
    )
    verdict = funnel_evaluator.evaluate(inp)
    assert verdict.passed is True
    assert verdict.stage == FunnelStage.PASSED
    assert verdict.playbook == PlaybookType.CROSS_ASSET_LEAD_LAG_CATCHUP
    assert verdict.allowed_side == TradeSide.LONG
    assert verdict.rejection_reason is None


def test_funnel_classifies_general_alpha_momentum(
    funnel_evaluator: TieredFunnelEvaluator,
) -> None:
    base = make_valid_trend_input()
    # Fails trend (hurst 0.45 < 0.50), fails reversal (no liq spike), fails lead-lag (symbol is BTC)
    # But max_prob 0.59 >= 0.52 -> General Alpha Momentum
    inp = TieredFunnelInput(
        symbol="BTCUSDT",
        spread_bps=base.spread_bps,
        yang_zhang_vol=base.yang_zhang_vol,
        garman_klass_vol=base.garman_klass_vol,
        is_blackout=False,
        is_account_frozen=False,
        p_long_cal=0.59,
        p_short_cal=0.41,
        uncertainty_width=base.uncertainty_width,
        is_ood=False,
        hurst_exp=0.45,
        oi_quadrant=OIPriceQuadrant.NEUTRAL,
        vpin_percentile=60.0,
        liq_long_z=0.1,
        liq_short_z=0.1,
        funding_z=0.0,
        vpin_slope_3bars=0.0,
        btc_15m_return_z=0.0,
        btc_lead_beta=1.0,
        lead_lag_catchup_gap_z=0.0,
    )
    verdict = funnel_evaluator.evaluate(inp)
    assert verdict.passed is True
    assert verdict.stage == FunnelStage.PASSED
    assert verdict.playbook == PlaybookType.GENERAL_ALPHA_MOMENTUM
    assert verdict.allowed_side == TradeSide.LONG
    assert verdict.rejection_reason is None


def test_funnel_rejects_on_stage_c_no_playbook(
    funnel_evaluator: TieredFunnelEvaluator,
) -> None:
    base = make_valid_trend_input()
    # Passes Stage A, passes Stage B (if hurdle low enough or edge just clear)
    # But prob 0.519 < 0.52 and no playbooks met
    # To test Stage C rejection specifically, let's configure edge to pass Stage B
    # but fail Stage C:
    # E.g. If fee gate min_calibrated_prob was met, but playbook validator threshold is higher:
    # Here both have min_calibrated_prob=0.52. If max_prob < 0.52, Stage B rejects it.
    # What if max_prob >= 0.52? classify_and_validate falls back to GENERAL_ALPHA_MOMENTUM!
    # Wait, can Stage C reject if e.g. a specific playbook condition fails without general alpha?
    # classify_and_validate will return GENERAL_ALPHA_MOMENTUM if max_prob >= cfg.trend_min_calibrated_prob.
    # If a custom playbook validator had trend_min_calibrated_prob higher than fee gate min_calibrated_prob:
    cfg = PlaybookConfig(
        trend_hurst_min=0.55,
        trend_vpin_percentile_max=88.0,
        trend_min_calibrated_prob=0.60,  # Higher than fee gate 0.52
        reversal_hurst_max=0.45,
        reversal_liq_z_min=2.0,
        reversal_funding_z_abs_min=2.0,
        reversal_vpin_slope_max=0.0,
        lead_lag_btc_ret_z_min=1.5,
        lead_lag_btc_beta_min=0.50,
        lead_lag_catchup_gap_z_min=1.0,
        altcoin_symbols=["ETHUSDT"],
    )
    custom_validator = PlaybookValidator(cfg)
    strict_funnel = TieredFunnelEvaluator(
        fee_to_edge_service=funnel_evaluator.fee_to_edge_service,
        playbook_validator=custom_validator,
        max_spread_bps=5.0,
        min_yz_vol=0.002,
        max_uncertainty_width=0.15,
    )
    inp = TieredFunnelInput(
        symbol="BTCUSDT",
        spread_bps=base.spread_bps,
        yang_zhang_vol=base.yang_zhang_vol,
        garman_klass_vol=base.garman_klass_vol,
        is_blackout=False,
        is_account_frozen=False,
        p_long_cal=0.55,  # >= 0.52 (passes Stage B), but < 0.60 (fails Stage C)
        p_short_cal=0.45,
        uncertainty_width=base.uncertainty_width,
        is_ood=False,
        hurst_exp=0.50,  # < 0.55 (fails trend)
        oi_quadrant=OIPriceQuadrant.NEUTRAL,  # fails trend
        vpin_percentile=60.0,
        liq_long_z=0.1,  # fails reversal
        liq_short_z=0.1,
        funding_z=0.0,
        vpin_slope_3bars=0.0,
        btc_15m_return_z=0.0,
        btc_lead_beta=1.0,
        lead_lag_catchup_gap_z=0.0,
    )
    verdict = strict_funnel.evaluate(inp)
    assert verdict.passed is False
    assert verdict.stage == FunnelStage.STAGE_C_PLAYBOOK
    assert verdict.rejection_reason == "NO_PLAYBOOK_PRECONDITIONS_MET"
