"""Deterministic Risk Calculator — 100% Explicit Config Injection, Zero Default Params, Zero Hardcode."""

from __future__ import annotations

import math

from pydantic import BaseModel

from trading_bot.domain.entities.risk.account_state import (
    AccountPortfolioContext,
    ActivePositionSnapshot,
)


class RiskConfig(BaseModel):
    """All parameters must be explicitly injected (Zero Defaults)."""

    maker_fee_rate: float
    taker_fee_rate: float
    min_tp_to_cost_multiple: float
    min_net_rr: float
    kelly_fraction: float
    max_risk_per_trade_pct: float
    min_executable_risk_pct: float
    max_total_open_heat_pct: float
    max_margin_utilization_pct: float
    target_30m_yz_vol: float
    min_vol_floor: float
    max_vol_scalar: float
    uncertainty_penalty_weight: float
    prior_neutral_prob: float
    corr_scalar_zero_same_dir: float
    corr_scalar_one_same_dir: float
    corr_scalar_two_plus_same_dir: float
    daily_drawdown_halt_pct: float
    max_drawdown_freeze_pct: float
    min_barrier_distance_pct: float
    min_leverage: int
    max_leverage: int


class SizingResult(BaseModel):
    """Sizing result entity with explicit fields (Zero Defaults)."""

    approved: bool
    rejection_reason: str | None
    net_rr: float
    roundtrip_cost_pct: float
    risk_pct_of_nav: float
    risk_amount_usd: float
    position_notional_usd: float
    recommended_leverage: int


class DeterministicRiskCalculator:
    def __init__(self, config: RiskConfig) -> None:
        self.cfg: RiskConfig = config

    @staticmethod
    def _rejected(
        reason: str, net_rr: float, roundtrip_cost_pct: float, min_lev: int
    ) -> SizingResult:
        return SizingResult(
            approved=False,
            rejection_reason=reason,
            net_rr=net_rr,
            roundtrip_cost_pct=roundtrip_cost_pct,
            risk_pct_of_nav=0.0,
            risk_amount_usd=0.0,
            position_notional_usd=0.0,
            recommended_leverage=min_lev,
        )

    @staticmethod
    def compute_effective_open_heat_pct(
        positions: list[ActivePositionSnapshot],
        equity_nav_usd: float,
    ) -> float:
        """Compute effective open heat excluding positions with SL moved to breakeven."""
        if equity_nav_usd <= 0.0:
            return 0.0
        at_risk_sum = sum(
            pos.effective_at_risk_usd for pos in positions if not pos.is_sl_at_breakeven
        )
        return at_risk_sum / equity_nav_usd

    @staticmethod
    def compute_raw_open_heat_pct(
        positions: list[ActivePositionSnapshot],
        equity_nav_usd: float,
    ) -> float:
        """Compute raw open heat including all open positions regardless of SL status."""
        if equity_nav_usd <= 0.0:
            return 0.0
        return sum(pos.effective_at_risk_usd for pos in positions) / equity_nav_usd

    def compute_roundtrip_cost_pct(
        self,
        is_limit_entry: bool,
        estimated_oneway_slippage_pct: float,
    ) -> float:
        if is_limit_entry:
            entry_fee = self.cfg.maker_fee_rate
            entry_slip = 0.0
        else:
            entry_fee = self.cfg.taker_fee_rate
            entry_slip = estimated_oneway_slippage_pct
        exit_fee = self.cfg.taker_fee_rate
        exit_slip = estimated_oneway_slippage_pct
        return entry_fee + exit_fee + entry_slip + exit_slip

    def _resolve_correlation_scalar(self, same_direction_open_count: int) -> float:
        if same_direction_open_count >= 2:
            return self.cfg.corr_scalar_two_plus_same_dir
        if same_direction_open_count == 1:
            return self.cfg.corr_scalar_one_same_dir
        return self.cfg.corr_scalar_zero_same_dir

    def evaluate_and_size_trade(
        self,
        nav_usd: float,
        current_daily_dd_pct: float,
        current_max_dd_pct: float,
        effective_open_heat_pct: float,
        same_direction_open_count: int,
        entry_price: float,
        stop_loss_price: float,
        take_profit_price: float,
        is_limit_entry: bool,
        kyle_slippage_pct: float,
        calibrated_win_prob: float,
        venn_abers_uncertainty_width: float,
        ood_shrinkage_factor: float,
        current_yz_vol: float,
        pre_event_blackout: bool,
        available_margin_usd: float,
        used_margin_usd: float,
        arbiter_size_multiplier: float,
    ) -> SizingResult:
        min_lev = self.cfg.min_leverage
        if current_max_dd_pct >= self.cfg.max_drawdown_freeze_pct:
            return self._rejected("MAX_DRAWDOWN_CIRCUIT_BREAKER_ACTIVE", 0.0, 0.0, min_lev)
        if current_daily_dd_pct >= self.cfg.daily_drawdown_halt_pct:
            return self._rejected("DAILY_DRAWDOWN_CIRCUIT_BREAKER_ACTIVE", 0.0, 0.0, min_lev)
        if pre_event_blackout:
            return self._rejected("MACRO_EVENT_BLACKOUT_WINDOW", 0.0, 0.0, min_lev)

        raw_sl_pct = abs(entry_price - stop_loss_price) / entry_price
        raw_tp_pct = abs(take_profit_price - entry_price) / entry_price
        if (
            raw_sl_pct <= self.cfg.min_barrier_distance_pct
            or raw_tp_pct <= self.cfg.min_barrier_distance_pct
        ):
            return self._rejected("INVALID_SL_OR_TP_DISTANCE", 0.0, 0.0, min_lev)

        cost_pct = self.compute_roundtrip_cost_pct(is_limit_entry, kyle_slippage_pct)
        min_required_tp = self.cfg.min_tp_to_cost_multiple * cost_pct
        if raw_tp_pct < min_required_tp:
            return self._rejected(
                f"TP_TOO_SMALL_VS_FEES ({raw_tp_pct:.4f} < {min_required_tp:.4f})",
                0.0,
                cost_pct,
                min_lev,
            )

        net_tp_pct = raw_tp_pct - cost_pct
        net_sl_pct = raw_sl_pct + cost_pct
        net_rr = net_tp_pct / net_sl_pct
        if net_rr < self.cfg.min_net_rr:
            return self._rejected(
                f"INSUFFICIENT_NET_RR ({net_rr:.2f} < {self.cfg.min_net_rr:.2f})",
                net_rr,
                cost_pct,
                min_lev,
            )

        p_base = self.cfg.prior_neutral_prob
        clamped_shrinkage = max(0.0, min(1.0, ood_shrinkage_factor))
        p_eff = p_base + (calibrated_win_prob - p_base) * clamped_shrinkage
        q_eff = 1.0 - p_eff
        full_kelly = (p_eff * net_rr - q_eff) / net_rr
        if full_kelly <= 0.0:
            return self._rejected(
                f"NON_POSITIVE_KELLY_EXPECTANCY (p_eff={p_eff:.3f}, net_rr={net_rr:.2f})",
                net_rr,
                cost_pct,
                min_lev,
            )

        uncertainty_penalty = max(
            0.0, 1.0 - self.cfg.uncertainty_penalty_weight * venn_abers_uncertainty_width
        )
        safe_vol = max(current_yz_vol, self.cfg.min_vol_floor)
        vol_scalar = min(self.cfg.max_vol_scalar, self.cfg.target_30m_yz_vol / safe_vol)
        clamped_arbiter_mult = max(0.0, min(1.0, arbiter_size_multiplier))
        raw_risk_pct = (
            self.cfg.kelly_fraction
            * full_kelly
            * uncertainty_penalty
            * vol_scalar
            * clamped_arbiter_mult
        )

        corr_multiplier = self._resolve_correlation_scalar(same_direction_open_count)
        capped_risk_pct = min(
            self.cfg.max_risk_per_trade_pct * corr_multiplier * clamped_arbiter_mult,
            raw_risk_pct,
        )
        remaining_heat_pct = max(0.0, self.cfg.max_total_open_heat_pct - effective_open_heat_pct)

        final_risk_pct = min(capped_risk_pct, remaining_heat_pct)

        if final_risk_pct < self.cfg.min_executable_risk_pct:
            return self._rejected(
                "PORTFOLIO_HEAT_LIMIT_REACHED_OR_SIZE_TOO_SMALL",
                net_rr,
                cost_pct,
                min_lev,
            )

        risk_usd = nav_usd * final_risk_pct
        notional_usd = risk_usd / net_sl_pct
        max_allowed_notional = nav_usd * self.cfg.max_leverage
        if notional_usd > max_allowed_notional:
            notional_usd = max_allowed_notional
            risk_usd = notional_usd * net_sl_pct
            final_risk_pct = risk_usd / nav_usd

        raw_lev = math.ceil(notional_usd / nav_usd)
        rec_leverage = max(self.cfg.min_leverage, min(self.cfg.max_leverage, raw_lev))

        # Margin capacity validation & capping against available margin and max utilization
        max_margin_capacity_usd = min(
            available_margin_usd,
            max(0.0, self.cfg.max_margin_utilization_pct * nav_usd - used_margin_usd),
        )
        if max_margin_capacity_usd <= 0.0:
            return self._rejected("INSUFFICIENT_AVAILABLE_MARGIN", net_rr, cost_pct, min_lev)

        required_margin_usd = notional_usd / rec_leverage
        if required_margin_usd > max_margin_capacity_usd:
            notional_usd = max_margin_capacity_usd * rec_leverage
            risk_usd = notional_usd * net_sl_pct
            final_risk_pct = risk_usd / nav_usd
            if final_risk_pct < self.cfg.min_executable_risk_pct:
                return self._rejected("INSUFFICIENT_AVAILABLE_MARGIN", net_rr, cost_pct, min_lev)

        return SizingResult(
            approved=True,
            rejection_reason=None,
            net_rr=round(net_rr, 3),
            roundtrip_cost_pct=round(cost_pct, 5),
            risk_pct_of_nav=round(final_risk_pct, 5),
            risk_amount_usd=round(risk_usd, 2),
            position_notional_usd=round(notional_usd, 2),
            recommended_leverage=rec_leverage,
        )

    def evaluate_with_portfolio_context(
        self,
        context: AccountPortfolioContext,
        arbiter_size_multiplier: float,
        entry_price: float,
        stop_loss_price: float,
        take_profit_price: float,
        is_limit_entry: bool,
        kyle_slippage_pct: float,
        calibrated_win_prob: float,
        venn_abers_uncertainty_width: float,
        ood_shrinkage_factor: float,
        current_yz_vol: float,
        pre_event_blackout: bool,
    ) -> SizingResult:
        """Evaluate trade using full AccountPortfolioContext snapshot."""
        return self.evaluate_and_size_trade(
            nav_usd=context.equity_nav_usd,
            current_daily_dd_pct=context.current_daily_dd_pct,
            current_max_dd_pct=context.current_max_dd_pct,
            effective_open_heat_pct=context.effective_open_heat_pct,
            same_direction_open_count=context.same_direction_active_count,
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            is_limit_entry=is_limit_entry,
            kyle_slippage_pct=kyle_slippage_pct,
            calibrated_win_prob=calibrated_win_prob,
            venn_abers_uncertainty_width=venn_abers_uncertainty_width,
            ood_shrinkage_factor=ood_shrinkage_factor,
            current_yz_vol=current_yz_vol,
            pre_event_blackout=pre_event_blackout,
            available_margin_usd=context.available_margin_usd,
            used_margin_usd=context.used_margin_usd,
            arbiter_size_multiplier=arbiter_size_multiplier,
        )
