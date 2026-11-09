"""Tiered Funnel Evaluator — Application Use Case.

Evaluates signals through a 3-stage exclusionary funnel:
- Stage A: Microstructure & Account Circuit Breakers (spread, Yang-Zhang vol, blackout, account freeze)
- Stage B: Fee-to-Edge & ML Calibration (OOD, Venn-Abers uncertainty, Garman-Klass expected edge)
- Stage C: Tactical Playbook Preconditions & Classification (trend, reversal, lead-lag, general alpha)

Enforces Invariant #9 (Zero Defaults, Zero Class Field Defaults, Zero Silent Fallbacks).
Clean Architecture: Application layer depends ONLY on Domain and Ports.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel

from trading_bot.domain.entities.common.enums import (
    OIPriceQuadrant,
    PlaybookType,
    TradeSide,
)
from trading_bot.domain.services.execution.fee_to_edge_gate import FeeToEdgeGatingService
from trading_bot.domain.services.execution.playbook_validator import (
    PlaybookPreconditionsInput,
    PlaybookValidator,
)


class FunnelStage(StrEnum):
    """Funnel execution stages."""

    STAGE_A_RULES = "STAGE_A_RULES"
    STAGE_B_FEE_EDGE = "STAGE_B_FEE_EDGE"
    STAGE_C_PLAYBOOK = "STAGE_C_PLAYBOOK"
    PASSED = "PASSED"


class TieredFunnelInput(BaseModel):
    """Input parameters for tiered signal gating evaluation (Zero Defaults)."""

    symbol: str
    spread_bps: float
    yang_zhang_vol: float
    garman_klass_vol: float
    is_blackout: bool
    is_account_frozen: bool
    p_long_cal: float
    p_short_cal: float
    uncertainty_width: float
    is_ood: bool
    hurst_exp: float
    oi_quadrant: OIPriceQuadrant
    vpin_percentile: float
    liq_long_z: float
    liq_short_z: float
    funding_z: float
    vpin_slope_3bars: float
    btc_15m_return_z: float
    btc_lead_beta: float
    lead_lag_catchup_gap_z: float


class TieredFunnelVerdict(BaseModel):
    """Verdict from tiered signal funnel evaluation (Zero Defaults)."""

    passed: bool
    stage: FunnelStage
    playbook: PlaybookType
    allowed_side: TradeSide
    expected_edge_bps: float
    hurdle_bps: float
    rejection_reason: str | None


class TieredFunnelEvaluator:
    """Evaluates market state through 3 tiered exclusionary gates."""

    def __init__(
        self,
        fee_to_edge_service: FeeToEdgeGatingService,
        playbook_validator: PlaybookValidator,
        max_spread_bps: float,
        min_yz_vol: float,
        max_uncertainty_width: float,
    ) -> None:
        self.fee_to_edge_service: FeeToEdgeGatingService = fee_to_edge_service
        self.playbook_validator: PlaybookValidator = playbook_validator
        self.max_spread_bps: float = max_spread_bps
        self.min_yz_vol: float = min_yz_vol
        self.max_uncertainty_width: float = max_uncertainty_width

    def evaluate(self, inp: TieredFunnelInput) -> TieredFunnelVerdict:
        """Evaluate a symbol input through the 3-stage exclusionary funnel."""
        # Stage A: Microstructure & Account Circuit Breakers
        if inp.is_account_frozen:
            return TieredFunnelVerdict(
                passed=False,
                stage=FunnelStage.STAGE_A_RULES,
                playbook=PlaybookType.NO_TRADE_REGIME,
                allowed_side=TradeSide.FLAT,
                expected_edge_bps=0.0,
                hurdle_bps=0.0,
                rejection_reason="ACCOUNT_FROZEN",
            )

        if inp.is_blackout:
            return TieredFunnelVerdict(
                passed=False,
                stage=FunnelStage.STAGE_A_RULES,
                playbook=PlaybookType.NO_TRADE_REGIME,
                allowed_side=TradeSide.FLAT,
                expected_edge_bps=0.0,
                hurdle_bps=0.0,
                rejection_reason="MACRO_EVENT_BLACKOUT",
            )

        if inp.spread_bps > self.max_spread_bps:
            return TieredFunnelVerdict(
                passed=False,
                stage=FunnelStage.STAGE_A_RULES,
                playbook=PlaybookType.NO_TRADE_REGIME,
                allowed_side=TradeSide.FLAT,
                expected_edge_bps=0.0,
                hurdle_bps=0.0,
                rejection_reason=f"SPREAD_TOO_WIDE ({inp.spread_bps:.2f} > {self.max_spread_bps:.2f})",
            )

        if inp.yang_zhang_vol < self.min_yz_vol:
            return TieredFunnelVerdict(
                passed=False,
                stage=FunnelStage.STAGE_A_RULES,
                playbook=PlaybookType.NO_TRADE_REGIME,
                allowed_side=TradeSide.FLAT,
                expected_edge_bps=0.0,
                hurdle_bps=0.0,
                rejection_reason=f"YZ_VOL_BELOW_MIN ({inp.yang_zhang_vol:.5f} < {self.min_yz_vol:.5f})",
            )

        # Stage B: Fee-to-Edge & ML Calibration
        if inp.is_ood:
            return TieredFunnelVerdict(
                passed=False,
                stage=FunnelStage.STAGE_B_FEE_EDGE,
                playbook=PlaybookType.NO_TRADE_REGIME,
                allowed_side=TradeSide.FLAT,
                expected_edge_bps=0.0,
                hurdle_bps=0.0,
                rejection_reason="OUT_OF_DISTRIBUTION_WARNING",
            )

        if inp.uncertainty_width > self.max_uncertainty_width:
            return TieredFunnelVerdict(
                passed=False,
                stage=FunnelStage.STAGE_B_FEE_EDGE,
                playbook=PlaybookType.NO_TRADE_REGIME,
                allowed_side=TradeSide.FLAT,
                expected_edge_bps=0.0,
                hurdle_bps=0.0,
                rejection_reason=f"UNCERTAINTY_TOO_WIDE ({inp.uncertainty_width:.4f} > {self.max_uncertainty_width:.4f})",
            )

        max_prob = max(inp.p_long_cal, inp.p_short_cal)
        fee_verdict = self.fee_to_edge_service.evaluate(
            p_cal=max_prob,
            garman_klass_vol=inp.garman_klass_vol,
        )

        if not fee_verdict.is_cleared:
            return TieredFunnelVerdict(
                passed=False,
                stage=FunnelStage.STAGE_B_FEE_EDGE,
                playbook=PlaybookType.NO_TRADE_REGIME,
                allowed_side=TradeSide.FLAT,
                expected_edge_bps=fee_verdict.expected_edge_bps,
                hurdle_bps=fee_verdict.hurdle_bps,
                rejection_reason=fee_verdict.reason,
            )

        # Stage C: Playbook Tactical Classification & Preconditions
        pb_input = PlaybookPreconditionsInput(
            symbol=inp.symbol,
            hurst_exp=inp.hurst_exp,
            oi_quadrant=inp.oi_quadrant,
            vpin_percentile=inp.vpin_percentile,
            p_long_cal=inp.p_long_cal,
            p_short_cal=inp.p_short_cal,
            liq_long_z=inp.liq_long_z,
            liq_short_z=inp.liq_short_z,
            funding_z=inp.funding_z,
            vpin_slope_3bars=inp.vpin_slope_3bars,
            btc_15m_return_z=inp.btc_15m_return_z,
            btc_lead_beta=inp.btc_lead_beta,
            lead_lag_catchup_gap_z=inp.lead_lag_catchup_gap_z,
            is_ood_warning=inp.is_ood,
            pre_event_blackout=inp.is_blackout,
        )
        pb_result = self.playbook_validator.classify_and_validate(pb_input)

        if not pb_result.is_valid:
            return TieredFunnelVerdict(
                passed=False,
                stage=FunnelStage.STAGE_C_PLAYBOOK,
                playbook=pb_result.playbook,
                allowed_side=pb_result.allowed_side,
                expected_edge_bps=fee_verdict.expected_edge_bps,
                hurdle_bps=fee_verdict.hurdle_bps,
                rejection_reason=pb_result.reason,
            )

        return TieredFunnelVerdict(
            passed=True,
            stage=FunnelStage.PASSED,
            playbook=pb_result.playbook,
            allowed_side=pb_result.allowed_side,
            expected_edge_bps=fee_verdict.expected_edge_bps,
            hurdle_bps=fee_verdict.hurdle_bps,
            rejection_reason=None,
        )
