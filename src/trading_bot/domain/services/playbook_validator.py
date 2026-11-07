"""Playbook Validator — Boundary condition verification for 4 Pre-Verified Playbooks."""

from __future__ import annotations

from pydantic import BaseModel

from trading_bot.domain.entities.enums import OIPriceQuadrant, PlaybookType, TradeSide


class PlaybookConfig(BaseModel):
    """Explicit threshold configuration for Playbook verification (Zero Defaults)."""

    trend_hurst_min: float
    trend_vpin_percentile_max: float
    trend_min_calibrated_prob: float
    reversal_hurst_max: float
    reversal_liq_z_min: float
    reversal_funding_z_abs_min: float
    reversal_vpin_slope_max: float
    lead_lag_btc_ret_z_min: float
    lead_lag_btc_beta_min: float
    lead_lag_catchup_gap_z_min: float
    altcoin_symbols: list[str]


class PlaybookPreconditionsInput(BaseModel):
    """Input features and market state required to validate playbooks (Zero Defaults)."""

    symbol: str
    hurst_exp: float
    oi_quadrant: OIPriceQuadrant
    vpin_percentile: float
    p_long_cal: float
    p_short_cal: float
    liq_long_z: float
    liq_short_z: float
    funding_z: float
    vpin_slope_3bars: float
    btc_15m_return_z: float
    btc_lead_beta: float
    lead_lag_catchup_gap_z: float
    is_ood_warning: bool
    pre_event_blackout: bool


class PlaybookValidationResult(BaseModel):
    """Validation result detailing approval and allowable trading direction (Zero Defaults)."""

    is_valid: bool
    playbook: PlaybookType
    allowed_side: TradeSide
    reason: str


class PlaybookValidator:
    def __init__(self, config: PlaybookConfig) -> None:
        self.cfg: PlaybookConfig = config

    def verify_playbook_preconditions(
        self,
        playbook: PlaybookType,
        market_input: PlaybookPreconditionsInput,
    ) -> PlaybookValidationResult:
        # Global circuit breakers enforce NO_TRADE_REGIME immediately
        if market_input.is_ood_warning:
            return PlaybookValidationResult(
                is_valid=False,
                playbook=PlaybookType.NO_TRADE_REGIME,
                allowed_side=TradeSide.FLAT,
                reason="OUT_OF_DISTRIBUTION_WARNING_ACTIVE",
            )
        if market_input.pre_event_blackout:
            return PlaybookValidationResult(
                is_valid=False,
                playbook=PlaybookType.NO_TRADE_REGIME,
                allowed_side=TradeSide.FLAT,
                reason="MACRO_EVENT_BLACKOUT_WINDOW_ACTIVE",
            )

        if playbook == PlaybookType.TREND_CONTINUATION_BREAKOUT:
            return self._verify_trend_continuation(market_input)
        if playbook == PlaybookType.LIQUIDATION_EXHAUSTION_REVERSAL:
            return self._verify_liquidation_exhaustion(market_input)
        if playbook == PlaybookType.CROSS_ASSET_LEAD_LAG_CATCHUP:
            return self._verify_lead_lag_catchup(market_input)

        return PlaybookValidationResult(
            is_valid=True,
            playbook=PlaybookType.NO_TRADE_REGIME,
            allowed_side=TradeSide.FLAT,
            reason="NO_TRADE_DEFAULT",
        )

    def _verify_trend_continuation(
        self,
        inp: PlaybookPreconditionsInput,
    ) -> PlaybookValidationResult:
        if inp.hurst_exp < self.cfg.trend_hurst_min:
            return PlaybookValidationResult(
                is_valid=False,
                playbook=PlaybookType.TREND_CONTINUATION_BREAKOUT,
                allowed_side=TradeSide.FLAT,
                reason=f"HURST_TOO_LOW ({inp.hurst_exp:.3f} < {self.cfg.trend_hurst_min:.3f})",
            )

        valid_oi = inp.oi_quadrant in (
            OIPriceQuadrant.LONG_BUILDUP,
            OIPriceQuadrant.SHORT_BUILDUP,
        )
        if not valid_oi:
            return PlaybookValidationResult(
                is_valid=False,
                playbook=PlaybookType.TREND_CONTINUATION_BREAKOUT,
                allowed_side=TradeSide.FLAT,
                reason=f"INVALID_OI_QUADRANT_FOR_TREND ({inp.oi_quadrant.value})",
            )

        if inp.vpin_percentile > self.cfg.trend_vpin_percentile_max:
            return PlaybookValidationResult(
                is_valid=False,
                playbook=PlaybookType.TREND_CONTINUATION_BREAKOUT,
                allowed_side=TradeSide.FLAT,
                reason=f"VPIN_TOXICITY_TOO_HIGH ({inp.vpin_percentile:.1f} > {self.cfg.trend_vpin_percentile_max:.1f})",
            )

        max_prob = max(inp.p_long_cal, inp.p_short_cal)
        if max_prob < self.cfg.trend_min_calibrated_prob:
            return PlaybookValidationResult(
                is_valid=False,
                playbook=PlaybookType.TREND_CONTINUATION_BREAKOUT,
                allowed_side=TradeSide.FLAT,
                reason=f"ML_CONFIDENCE_INSUFFICIENT ({max_prob:.3f} < {self.cfg.trend_min_calibrated_prob:.3f})",
            )

        if inp.oi_quadrant == OIPriceQuadrant.LONG_BUILDUP and inp.p_long_cal >= inp.p_short_cal:
            side = TradeSide.LONG
        elif inp.oi_quadrant == OIPriceQuadrant.SHORT_BUILDUP and inp.p_short_cal >= inp.p_long_cal:
            side = TradeSide.SHORT
        else:
            return PlaybookValidationResult(
                is_valid=False,
                playbook=PlaybookType.TREND_CONTINUATION_BREAKOUT,
                allowed_side=TradeSide.FLAT,
                reason="OI_DIRECTION_AND_ML_DIRECTION_MISMATCH",
            )

        return PlaybookValidationResult(
            is_valid=True,
            playbook=PlaybookType.TREND_CONTINUATION_BREAKOUT,
            allowed_side=side,
            reason="TREND_CONTINUATION_PRECONDITIONS_SATISFIED",
        )

    def _verify_liquidation_exhaustion(
        self,
        inp: PlaybookPreconditionsInput,
    ) -> PlaybookValidationResult:
        if inp.hurst_exp > self.cfg.reversal_hurst_max:
            return PlaybookValidationResult(
                is_valid=False,
                playbook=PlaybookType.LIQUIDATION_EXHAUSTION_REVERSAL,
                allowed_side=TradeSide.FLAT,
                reason=f"HURST_TOO_HIGH_FOR_REVERSAL ({inp.hurst_exp:.3f} > {self.cfg.reversal_hurst_max:.3f})",
            )

        has_liq_spike = (
            inp.liq_long_z >= self.cfg.reversal_liq_z_min
            or inp.liq_short_z >= self.cfg.reversal_liq_z_min
            or abs(inp.funding_z) >= self.cfg.reversal_funding_z_abs_min
        )
        if not has_liq_spike:
            return PlaybookValidationResult(
                is_valid=False,
                playbook=PlaybookType.LIQUIDATION_EXHAUSTION_REVERSAL,
                allowed_side=TradeSide.FLAT,
                reason="NO_SIGNIFICANT_LIQUIDATION_OR_FUNDING_SPIKE",
            )

        if inp.vpin_slope_3bars > self.cfg.reversal_vpin_slope_max:
            return PlaybookValidationResult(
                is_valid=False,
                playbook=PlaybookType.LIQUIDATION_EXHAUSTION_REVERSAL,
                allowed_side=TradeSide.FLAT,
                reason=f"VPIN_STILL_ACCELERATING ({inp.vpin_slope_3bars:.3f} > {self.cfg.reversal_vpin_slope_max:.3f})",
            )

        # Counter-trade the liquidation exhaustion
        if inp.liq_long_z >= self.cfg.reversal_liq_z_min or inp.funding_z <= (
            -self.cfg.reversal_funding_z_abs_min
        ):
            side = TradeSide.LONG
        else:
            side = TradeSide.SHORT

        return PlaybookValidationResult(
            is_valid=True,
            playbook=PlaybookType.LIQUIDATION_EXHAUSTION_REVERSAL,
            allowed_side=side,
            reason="LIQUIDATION_EXHAUSTION_PRECONDITIONS_SATISFIED",
        )

    def _verify_lead_lag_catchup(
        self,
        inp: PlaybookPreconditionsInput,
    ) -> PlaybookValidationResult:
        if inp.symbol not in self.cfg.altcoin_symbols:
            return PlaybookValidationResult(
                is_valid=False,
                playbook=PlaybookType.CROSS_ASSET_LEAD_LAG_CATCHUP,
                allowed_side=TradeSide.FLAT,
                reason=f"SYMBOL_NOT_AN_ALTCOIN ({inp.symbol})",
            )

        if abs(inp.btc_15m_return_z) < self.cfg.lead_lag_btc_ret_z_min:
            return PlaybookValidationResult(
                is_valid=False,
                playbook=PlaybookType.CROSS_ASSET_LEAD_LAG_CATCHUP,
                allowed_side=TradeSide.FLAT,
                reason=f"BTC_RETURN_Z_INSUFFICIENT ({abs(inp.btc_15m_return_z):.2f} < {self.cfg.lead_lag_btc_ret_z_min:.2f})",
            )

        if inp.btc_lead_beta < self.cfg.lead_lag_btc_beta_min:
            return PlaybookValidationResult(
                is_valid=False,
                playbook=PlaybookType.CROSS_ASSET_LEAD_LAG_CATCHUP,
                allowed_side=TradeSide.FLAT,
                reason=f"BTC_BETA_TOO_LOW ({inp.btc_lead_beta:.2f} < {self.cfg.lead_lag_btc_beta_min:.2f})",
            )

        if abs(inp.lead_lag_catchup_gap_z) < self.cfg.lead_lag_catchup_gap_z_min:
            return PlaybookValidationResult(
                is_valid=False,
                playbook=PlaybookType.CROSS_ASSET_LEAD_LAG_CATCHUP,
                allowed_side=TradeSide.FLAT,
                reason=f"CATCHUP_GAP_TOO_SMALL ({abs(inp.lead_lag_catchup_gap_z):.2f} < {self.cfg.lead_lag_catchup_gap_z_min:.2f})",
            )

        side = TradeSide.LONG if inp.btc_15m_return_z > 0.0 else TradeSide.SHORT
        return PlaybookValidationResult(
            is_valid=True,
            playbook=PlaybookType.CROSS_ASSET_LEAD_LAG_CATCHUP,
            allowed_side=side,
            reason="LEAD_LAG_CATCHUP_PRECONDITIONS_SATISFIED",
        )
