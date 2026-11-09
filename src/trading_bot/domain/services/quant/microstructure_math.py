"""Microstructure Mathematical Domain Services (Zero Defaults).

Pure Python implementation of high-frequency orderbook and trade flow microstructure:
- VPIN (Volume-Synchronized Probability of Informed Trading)
- Intra-Bar Kyle's Lambda (Price impact and execution slippage forecasting)
- Volatility-Adjusted Amihud Illiquidity (VA-ILLIQ)
- Corwin-Schultz High-Low Bid-Ask Spread Estimator
- Microstructure Noise Variance Ratio (MNVR)

Strictly adheres to Clean Architecture: 100% Pure Python (stdlib math only),
zero numpy, zero polars, zero numba, and zero default arguments (Invariant #9).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from trading_bot.domain.entities.market.market_state import (
    EnrichedMicroBar1m,
    MicroBar1m,
    MultiTFBar,
)
from trading_bot.domain.entities.quant.quant_configs import VPINConfig
from trading_bot.domain.exceptions import (
    InsufficientDataException,
    NumericalInstabilityException,
    ZeroVolumeException,
)


class MicrostructureMathService:
    """Pure domain service for deterministic market microstructure calculations."""

    def compute_vpin(
        self,
        bars_1m: Sequence[MicroBar1m | EnrichedMicroBar1m],
        adv_usd: float,
        cfg: VPINConfig,
    ) -> float:
        """Compute Volume-Synchronized Probability of Informed Trading (VPIN).

        Slices trade volume into equal-sized volume buckets V_bucket = ADV * slice_ratio.
        Classifies aggressive taker buy vs sell volume per bucket.
        Computes rolling imbalance across N buckets.
        """
        if adv_usd <= 0.0:
            raise ZeroVolumeException("Average Daily Volume (ADV) must be strictly positive.")
        if not bars_1m:
            raise InsufficientDataException("Micro bars sequence must not be empty.")
        if cfg.num_buckets <= 0:
            raise InsufficientDataException("num_buckets must be strictly positive.")
        if cfg.volume_bucket_slice_ratio <= 0.0:
            raise ZeroVolumeException("volume_bucket_slice_ratio must be strictly positive.")

        v_bucket = adv_usd * cfg.volume_bucket_slice_ratio
        if v_bucket <= cfg.division_eps:
            raise ZeroVolumeException(
                f"Volume bucket size {v_bucket} is below division epsilon {cfg.division_eps}."
            )

        completed_buckets: list[tuple[float, float]] = []
        current_bucket_vol = 0.0
        current_bucket_buy = 0.0

        for bar in bars_1m:
            # Determine total volume in USD
            if isinstance(bar, EnrichedMicroBar1m):
                bar_vol_usd = bar.quote_volume if bar.quote_volume > 0.0 else bar.volume * bar.close
                if bar.volume > 0.0:
                    taker_buy_ratio = min(1.0, max(0.0, bar.taker_buy_volume / bar.volume))
                    bar_buy_usd = bar_vol_usd * taker_buy_ratio
                else:
                    bar_buy_usd = bar_vol_usd * 0.5
            else:
                bar_vol_usd = bar.volume * bar.close
                # Bulk Volume Classification (BVC) proxy for basic OHLCV bars
                price_range = bar.high - bar.low
                if price_range > cfg.division_eps:
                    bvc_buy_ratio = min(1.0, max(0.0, (bar.close - bar.low) / price_range))
                else:
                    if bar.close > bar.open:
                        bvc_buy_ratio = 1.0
                    elif bar.close < bar.open:
                        bvc_buy_ratio = 0.0
                    else:
                        bvc_buy_ratio = 0.5
                bar_buy_usd = bar_vol_usd * bvc_buy_ratio

            if bar_vol_usd <= 0.0:
                continue

            buy_fraction = bar_buy_usd / bar_vol_usd
            rem_bar_vol = bar_vol_usd

            # Fill buckets incrementally
            while rem_bar_vol > 0.0:
                space_in_bucket = v_bucket - current_bucket_vol
                if rem_bar_vol >= space_in_bucket:
                    # Current bucket is filled
                    fill_vol = space_in_bucket
                    fill_buy = fill_vol * buy_fraction
                    current_bucket_buy += fill_buy
                    current_bucket_vol = v_bucket
                    v_sell = max(0.0, v_bucket - current_bucket_buy)
                    completed_buckets.append((current_bucket_buy, v_sell))
                    # Reset bucket for next iteration
                    current_bucket_vol = 0.0
                    current_bucket_buy = 0.0
                    rem_bar_vol -= fill_vol
                else:
                    # Partially fill bucket
                    fill_vol = rem_bar_vol
                    fill_buy = fill_vol * buy_fraction
                    current_bucket_vol += fill_vol
                    current_bucket_buy += fill_buy
                    rem_bar_vol = 0.0

        if not completed_buckets:
            raise InsufficientDataException(
                f"Cumulative volume is insufficient to complete even 1 bucket of size {v_bucket}."
            )

        # Take the most recent N buckets (causal backward-looking window)
        active_buckets = completed_buckets[-cfg.num_buckets :]
        num_eval_buckets = len(active_buckets)

        abs_imbalance_sum = sum(abs(b[0] - b[1]) for b in active_buckets)
        vpin = abs_imbalance_sum / (num_eval_buckets * v_bucket)

        if not math.isfinite(vpin):
            raise NumericalInstabilityException(
                f"VPIN calculation produced non-finite value: {vpin}"
            )

        return min(1.0, max(0.0, vpin))

    def compute_kyle_lambda(
        self,
        bars_1m: Sequence[MicroBar1m | EnrichedMicroBar1m],
        cfg: VPINConfig,
    ) -> tuple[float, float]:
        """Compute Intra-Bar Kyle's Lambda (lambda_t) and Estimated One-Way Slippage.

        Regresses absolute 1m price changes on net aggressive order flow (SV_i) in USD.
        Returns:
            tuple[float, float]: (kyle_lambda, estimated_oneway_slippage_pct)
        """
        if not bars_1m:
            raise InsufficientDataException("Micro bars sequence must not be empty.")

        sum_dp_sv = 0.0
        sum_sv_sq = 0.0

        for bar in bars_1m:
            open_denom = max(bar.open, cfg.division_eps)
            dp_i = abs(bar.close - bar.open) / open_denom

            if isinstance(bar, EnrichedMicroBar1m):
                total_vol_usd = (
                    bar.quote_volume if bar.quote_volume > 0.0 else bar.volume * bar.close
                )
                if bar.volume > 0.0:
                    taker_buy_usd = total_vol_usd * (bar.taker_buy_volume / bar.volume)
                else:
                    taker_buy_usd = total_vol_usd * 0.5
            else:
                total_vol_usd = bar.volume * bar.close
                price_range = bar.high - bar.low
                if price_range > cfg.division_eps:
                    ratio = (bar.close - bar.low) / price_range
                else:
                    ratio = 0.5
                taker_buy_usd = total_vol_usd * ratio

            # Signed order flow in USD
            sv_i = abs(2.0 * taker_buy_usd - total_vol_usd)

            sum_dp_sv += dp_i * sv_i
            sum_sv_sq += sv_i * sv_i

        denom = sum_sv_sq + cfg.division_eps
        kyle_lambda = max(0.0, sum_dp_sv / denom)

        if not math.isfinite(kyle_lambda):
            raise NumericalInstabilityException(
                f"Kyle's Lambda calculation produced non-finite value: {kyle_lambda}"
            )

        # Baseline execution slippage forecast for standard $10,000 order size and 1 bps floor
        standard_order_size_usd = 10000.0
        standard_spread_floor = 0.0001  # 1 basis point floor
        estimated_slippage_pct = max(
            standard_spread_floor, standard_spread_floor + kyle_lambda * standard_order_size_usd
        )

        return (kyle_lambda, estimated_slippage_pct)

    def estimate_execution_slippage(
        self,
        kyle_lambda: float,
        order_size_usd: float,
        spread_bps: float,
    ) -> float:
        """Estimate execution slippage percentage for a specific order size and spread."""
        if order_size_usd < 0.0:
            raise InsufficientDataException("order_size_usd must be non-negative.")
        spread_fraction = max(0.0, spread_bps) * 1e-4
        raw_slippage = 0.5 * spread_fraction + kyle_lambda * order_size_usd
        return max(0.0001, raw_slippage)

    def compute_va_amihud(
        self,
        bars_1m: Sequence[MicroBar1m | EnrichedMicroBar1m],
        yz_vol: float,
        cfg: VPINConfig,
    ) -> float:
        """Compute Volatility-Adjusted Amihud Illiquidity Ratio (VA-ILLIQ).

        Normalizes price impact per USD volume by instantaneous Yang-Zhang volatility.
        """
        if not bars_1m:
            raise InsufficientDataException("Micro bars sequence must not be empty.")

        n_bars = len(bars_1m)
        vol_scalar = max(abs(yz_vol), cfg.division_eps)
        total_impact = 0.0

        for bar in bars_1m:
            open_denom = max(bar.open, cfg.division_eps)
            ret_abs = abs(bar.close - bar.open) / open_denom

            if isinstance(bar, EnrichedMicroBar1m):
                vol_usd = bar.quote_volume if bar.quote_volume > 0.0 else bar.volume * bar.close
            else:
                vol_usd = bar.volume * bar.close

            vol_denom = vol_usd + cfg.division_eps
            total_impact += ret_abs / (vol_scalar * vol_denom)

        va_amihud = (total_impact / n_bars) * 1e6
        if not math.isfinite(va_amihud):
            raise NumericalInstabilityException(
                f"VA-Amihud calculation produced non-finite value: {va_amihud}"
            )
        return max(0.0, va_amihud)

    def compute_corwin_schultz_spread(
        self,
        bars_30m: Sequence[MultiTFBar],
        cfg: VPINConfig,
    ) -> float:
        """Compute Corwin-Schultz High-Low Bid-Ask Spread Estimator in basis points (bps).

        Reconstructs effective bid-ask spread directly from 30m High and Low prices.
        """
        if len(bars_30m) < 2:
            raise InsufficientDataException(
                f"Corwin-Schultz estimator requires at least 2 bars, received {len(bars_30m)}."
            )

        denom_const = 3.0 - 2.0 * math.sqrt(2.0)
        spread_bps_list: list[float] = []

        for i in range(1, len(bars_30m)):
            prev_bar = bars_30m[i - 1]
            curr_bar = bars_30m[i]

            h1 = max(prev_bar.high, cfg.division_eps)
            l1 = max(prev_bar.low, cfg.division_eps)
            h2 = max(curr_bar.high, cfg.division_eps)
            l2 = max(curr_bar.low, cfg.division_eps)

            if l1 <= 0.0 or l2 <= 0.0:
                continue

            hl_ratio_1 = max(h1 / l1, 1.0)
            hl_ratio_2 = max(h2 / l2, 1.0)

            beta = (math.log(hl_ratio_1)) ** 2 + (math.log(hl_ratio_2)) ** 2

            h_max = max(h1, h2)
            l_min = min(l1, l2)
            gamma = (math.log(max(h_max / l_min, 1.0))) ** 2

            alpha = (math.sqrt(2.0 * beta) - math.sqrt(beta)) / denom_const - math.sqrt(
                gamma / denom_const
            )

            # S = 2 * (exp(alpha) - 1) / (1 + exp(alpha))
            # Bound alpha to avoid exp overflow
            alpha_bounded = min(alpha, 20.0)
            exp_alpha = math.exp(alpha_bounded)
            raw_spread = 2.0 * (exp_alpha - 1.0) / (1.0 + exp_alpha)
            clamped_spread = max(0.0, raw_spread)
            spread_bps_list.append(clamped_spread * 10000.0)

        if not spread_bps_list:
            return 0.0

        mean_spread_bps = sum(spread_bps_list) / len(spread_bps_list)
        if not math.isfinite(mean_spread_bps):
            raise NumericalInstabilityException(
                f"Corwin-Schultz spread produced non-finite value: {mean_spread_bps}"
            )
        return max(0.0, mean_spread_bps)

    def compute_mnvr(
        self,
        rv_1m: float,
        rv_15m: float,
        cfg: VPINConfig,
    ) -> float:
        """Compute Microstructure Noise Variance Ratio (MNVR).

        Quantifies high-frequency noise bounce relative to fundamental variance.
        """
        denom = max(rv_1m, 0.0) + cfg.division_eps
        raw_ratio = (rv_1m - rv_15m) / denom
        mnvr = max(0.0, raw_ratio)
        if not math.isfinite(mnvr):
            raise NumericalInstabilityException(
                f"MNVR calculation produced non-finite value: {mnvr}"
            )
        return mnvr
