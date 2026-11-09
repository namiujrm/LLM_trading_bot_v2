"""Multi-Timeframe Market Regime and Statistical Physics Domain Services (Zero Defaults).

Pure Python implementation of statistical physics, range-based volatility,
discrete jump detection, and information theory entropy metrics:
- Anis-Lloyd Corrected Hurst Exponent (R/S analysis with small-sample bias removal)
- Yang-Zhang Minimum-Variance Volatility
- Garman-Klass Extreme-Value Volatility
- Parkinson Range-Based Volatility
- Realized Bipower Variation (BV) & Barndorff-Nielsen Jump Z-Score
- Shannon Return Entropy
- Multiscale Permutation Entropy (Bandt & Pompe)
- Tsallis Non-Extensive Power-Law Entropy
- Macroeconomic Regime Classification

Strict Clean Architecture: 100% Pure Python (stdlib math only), zero numpy,
zero polars, zero numba, and zero default arguments (Invariant #9).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from trading_bot.domain.entities.common.enums import MarketRegime
from trading_bot.domain.entities.market.market_state import MicroBar1m, MultiTFBar
from trading_bot.domain.entities.quant.quant_configs import RegimeConfig
from trading_bot.domain.exceptions import (
    InsufficientDataException,
    NumericalInstabilityException,
)


class RegimeDetectorService:
    """Pure domain service for deterministic multi-timeframe market regime detection."""

    def compute_anis_lloyd_hurst(
        self,
        returns_30m: list[float],
        cfg: RegimeConfig,
    ) -> float:
        """Compute Anis-Lloyd Bias-Corrected Hurst Exponent (H_t).

        Corrects the upward small-sample bias of classical Rescaled Range (R/S) analysis
        via the analytical expectation under Gaussian random walk.
        Returns:
            float: Hurst exponent H_t in [0.05, 0.95].
        """
        n_samples = len(returns_30m)
        if n_samples < 10:
            raise InsufficientDataException(
                f"Hurst exponent requires at least 10 returns, received {n_samples}."
            )

        # Select candidate sub-chunk segment sizes
        candidate_sizes = [10, 20, 25, 50, 100]
        valid_sub_sizes = [s for s in candidate_sizes if s <= n_samples // 2]
        if len(valid_sub_sizes) < 2:
            half = n_samples // 2
            quarter = n_samples // 4
            if quarter >= 4 and quarter not in valid_sub_sizes:
                valid_sub_sizes.append(quarter)
            if half >= 4 and half not in valid_sub_sizes:
                valid_sub_sizes.append(half)
            valid_sub_sizes.sort()

        if len(valid_sub_sizes) < 2:
            # Insufficient degrees of freedom for regression; return baseline 0.50
            return 0.50

        log_m_list: list[float] = []
        log_rs_diff_list: list[float] = []

        for m in valid_sub_sizes:
            num_blocks = n_samples // m
            rs_sum = 0.0
            valid_blocks = 0

            for b in range(num_blocks):
                block = returns_30m[b * m : (b + 1) * m]
                mean_b = sum(block) / m
                var_b = sum((x - mean_b) ** 2 for x in block) / m
                std_b = math.sqrt(var_b)

                if std_b <= cfg.division_eps:
                    continue

                # Cumulative deviations from mean
                cum_dev = 0.0
                min_dev = 0.0
                max_dev = 0.0
                for x in block:
                    cum_dev += x - mean_b
                    if cum_dev < min_dev:
                        min_dev = cum_dev
                    if cum_dev > max_dev:
                        max_dev = cum_dev

                r_range = max_dev - min_dev
                rs_sum += r_range / std_b
                valid_blocks += 1

            if valid_blocks == 0:
                continue

            rs_empirical = rs_sum / valid_blocks

            # Theoretical Anis-Lloyd Expectation E[R/S]_m
            if m <= 340:
                gamma_ratio = math.gamma((m - 1.0) / 2.0) / (
                    math.sqrt(math.pi) * math.gamma(m / 2.0)
                )
                sum_sqrt = sum(math.sqrt((m - i) / i) for i in range(1, m))
                e_rs = gamma_ratio * sum_sqrt
            else:
                sum_sqrt = sum(math.sqrt((m - i) / i) for i in range(1, m))
                e_rs = (1.0 / math.sqrt(math.pi * m / 2.0)) * sum_sqrt

            e_rs = max(e_rs, cfg.division_eps)
            diff = math.log(rs_empirical + cfg.division_eps) - math.log(e_rs)
            log_m_list.append(math.log(float(m)))
            log_rs_diff_list.append(diff)

        if len(log_m_list) < 2:
            return 0.50

        # Linear regression of log_rs_diff on log_m
        mean_x = sum(log_m_list) / len(log_m_list)
        mean_y = sum(log_rs_diff_list) / len(log_rs_diff_list)

        cov_xy = sum(
            (log_m_list[i] - mean_x) * (log_rs_diff_list[i] - mean_y)
            for i in range(len(log_m_list))
        )
        var_x = sum((x - mean_x) ** 2 for x in log_m_list)

        denom = var_x + cfg.division_eps
        slope_beta = cov_xy / denom

        # H = 0.50 + beta
        hurst = 0.50 + slope_beta
        if not math.isfinite(hurst):
            raise NumericalInstabilityException(
                f"Hurst exponent calculation produced non-finite value: {hurst}"
            )

        return min(0.95, max(0.05, hurst))

    def compute_yang_zhang_vol(
        self,
        bars_30m: list[MultiTFBar],
        cfg: RegimeConfig,
    ) -> float:
        """Compute Yang-Zhang Minimum-Variance Volatility (sigma_YZ).

        Combines overnight jump variance, continuous drift, and Rogers-Satchell range variance.
        """
        n_bars = len(bars_30m)
        if n_bars < 2:
            raise InsufficientDataException(
                f"Yang-Zhang volatility requires at least 2 bars, received {n_bars}."
            )

        m_trans = n_bars - 1
        o_returns: list[float] = []
        c_returns: list[float] = []
        rs_terms: list[float] = []

        for i in range(1, n_bars):
            prev_close = max(bars_30m[i - 1].close, cfg.division_eps)
            curr_open = max(bars_30m[i].open, cfg.division_eps)
            curr_high = max(bars_30m[i].high, cfg.division_eps)
            curr_low = max(bars_30m[i].low, cfg.division_eps)
            curr_close = max(bars_30m[i].close, cfg.division_eps)

            # Overnight jump: ln(Open_i / Close_{i-1})
            o_ret = math.log(curr_open / prev_close)
            # Continuous open-to-close: ln(Close_i / Open_i)
            c_ret = math.log(curr_close / curr_open)

            o_returns.append(o_ret)
            c_returns.append(c_ret)

            # Rogers-Satchell variance term
            u_i = math.log(curr_high / curr_open)
            d_i = math.log(curr_low / curr_open)
            c_i = math.log(curr_close / curr_open)
            rs = u_i * (u_i - c_i) + d_i * (d_i - c_i)
            rs_terms.append(max(0.0, rs))

        mean_o = sum(o_returns) / m_trans
        mean_c = sum(c_returns) / m_trans

        denom_m = max(m_trans - 1, 1)
        sigma_o_sq = sum((x - mean_o) ** 2 for x in o_returns) / denom_m
        sigma_c_sq = sum((x - mean_c) ** 2 for x in c_returns) / denom_m
        sigma_rs_sq = sum(rs_terms) / m_trans

        k_denom = 1.34 + (m_trans + 1.0) / denom_m
        k_weight = 0.34 / k_denom

        sigma_yz_sq = sigma_o_sq + k_weight * sigma_c_sq + (1.0 - k_weight) * sigma_rs_sq
        sigma_yz = math.sqrt(max(0.0, sigma_yz_sq))

        if not math.isfinite(sigma_yz):
            raise NumericalInstabilityException(
                f"Yang-Zhang calculation produced non-finite value: {sigma_yz}"
            )
        return sigma_yz

    def compute_garman_klass_vol(
        self,
        bars_30m: list[MultiTFBar],
        cfg: RegimeConfig,
    ) -> float:
        """Compute Garman-Klass Extreme-Value Volatility (sigma_GK)."""
        n_bars = len(bars_30m)
        if n_bars < 1:
            raise InsufficientDataException("Garman-Klass volatility requires at least 1 bar.")

        sum_gk = 0.0
        co_factor = 2.0 * math.log(2.0) - 1.0  # ~0.386294

        for bar in bars_30m:
            o_val = max(bar.open, cfg.division_eps)
            h_val = max(bar.high, cfg.division_eps)
            l_val = max(bar.low, cfg.division_eps)
            c_val = max(bar.close, cfg.division_eps)

            log_hl = math.log(max(h_val / l_val, 1.0))
            log_co = math.log(c_val / o_val)

            gk_bar = 0.5 * (log_hl**2) - co_factor * (log_co**2)
            sum_gk += max(0.0, gk_bar)

        sigma_gk = math.sqrt(sum_gk / n_bars)
        if not math.isfinite(sigma_gk):
            raise NumericalInstabilityException(
                f"Garman-Klass calculation produced non-finite value: {sigma_gk}"
            )
        return sigma_gk

    def compute_parkinson_vol(
        self,
        bars_30m: list[MultiTFBar],
        cfg: RegimeConfig,
    ) -> float:
        """Compute Parkinson Extreme-Value Volatility (sigma_P)."""
        n_bars = len(bars_30m)
        if n_bars < 1:
            raise InsufficientDataException("Parkinson volatility requires at least 1 bar.")

        sum_hl_sq = 0.0
        for bar in bars_30m:
            h_val = max(bar.high, cfg.division_eps)
            l_val = max(bar.low, cfg.division_eps)
            log_hl = math.log(max(h_val / l_val, 1.0))
            sum_hl_sq += log_hl**2

        denom = 4.0 * math.log(2.0) * n_bars
        sigma_p = math.sqrt(max(0.0, sum_hl_sq / denom))
        if not math.isfinite(sigma_p):
            raise NumericalInstabilityException(
                f"Parkinson calculation produced non-finite value: {sigma_p}"
            )
        return sigma_p

    def compute_bipower_variation_jump(
        self,
        bars_1m: Sequence[MicroBar1m],
        cfg: RegimeConfig,
    ) -> tuple[float, float, bool]:
        """Compute Realized Bipower Variation (BV) and Jump Z-Score.

        Decomposes realized variance into continuous diffusion and discrete jump components.
        Returns:
            tuple[float, float, bool]: (bipower_variation, jump_z_score, has_volatility_jump)
        """
        n_bars = len(bars_1m)
        if n_bars < 4:
            return (0.0, 0.0, False)

        m_returns = n_bars - 1
        returns: list[float] = []

        for i in range(1, n_bars):
            prev_c = max(bars_1m[i - 1].close, cfg.division_eps)
            curr_c = max(bars_1m[i].close, cfg.division_eps)
            returns.append(math.log(curr_c / prev_c))

        rv = sum(r**2 for r in returns)

        # Barndorff-Nielsen & Shephard mu_1 = sqrt(2 / pi)
        mu_1_inv2 = math.pi / 2.0  # mu_1^(-2)
        bv_sum = sum(abs(returns[i]) * abs(returns[i - 1]) for i in range(1, m_returns))
        bv = mu_1_inv2 * (m_returns / (m_returns - 1)) * bv_sum

        # Tri-Power Quarticity (TPQ)
        mu_43 = (2.0 ** (2.0 / 3.0)) * math.gamma(7.0 / 6.0) / math.gamma(0.5)
        mu_43_inv3 = 1.0 / (mu_43**3)

        tpq_sum = sum(
            (abs(returns[i]) ** (4.0 / 3.0))
            * (abs(returns[i - 1]) ** (4.0 / 3.0))
            * (abs(returns[i - 2]) ** (4.0 / 3.0))
            for i in range(2, m_returns)
        )
        tpq = m_returns * mu_43_inv3 * (m_returns / (m_returns - 2)) * tpq_sum

        # Asymptotic variance factor: c = mu_1^(-4) + 2*mu_1^(-2) - 5
        c_factor = (math.pi / 2.0) ** 2 + math.pi - 5.0  # ~0.609068

        if rv <= cfg.division_eps or bv <= cfg.division_eps:
            return (bv, 0.0, False)

        jump_ratio = (rv - bv) / (rv + cfg.division_eps)
        tpq_ratio = max(1.0, tpq / (bv**2 + cfg.division_eps))
        v_jump = c_factor * (1.0 / m_returns) * tpq_ratio

        jump_z_score = jump_ratio / math.sqrt(v_jump + cfg.division_eps)

        if not math.isfinite(jump_z_score):
            jump_z_score = 0.0

        has_jump = jump_z_score >= cfg.bipower_jump_confidence
        return (bv, jump_z_score, has_jump)

    def compute_shannon_entropy(
        self,
        returns: list[float],
        cfg: RegimeConfig,
    ) -> float:
        """Compute Normalized Shannon Information Entropy of returns distribution."""
        n_samples = len(returns)
        if n_samples == 0:
            raise InsufficientDataException("Returns list must not be empty.")

        k_bins = cfg.entropy_bins
        if k_bins <= 1:
            return 0.0

        min_val = min(returns)
        max_val = max(returns)
        val_range = max_val - min_val

        if val_range <= cfg.division_eps:
            return 0.0

        bin_width = val_range / k_bins
        bin_counts = [0] * k_bins

        for r in returns:
            idx = int((r - min_val) / bin_width)
            if idx >= k_bins:
                idx = k_bins - 1
            bin_counts[idx] += 1

        entropy = 0.0
        for count in bin_counts:
            if count > 0:
                p = count / n_samples
                entropy -= p * math.log(p)

        norm_entropy = entropy / math.log(float(k_bins))
        return min(1.0, max(0.0, norm_entropy))

    def compute_permutation_entropy(
        self,
        returns: list[float],
        cfg: RegimeConfig,
    ) -> float:
        """Compute Normalized Permutation Entropy (Bandt & Pompe 2002)."""
        m_dim = cfg.permutation_dimension
        n_samples = len(returns)
        if n_samples < m_dim:
            raise InsufficientDataException(
                f"Permutation entropy requires at least {m_dim} returns, received {n_samples}."
            )

        pattern_counts: dict[tuple[int, ...], int] = {}
        total_patterns = n_samples - m_dim + 1

        for i in range(total_patterns):
            sub_window = returns[i : i + m_dim]
            # Map values into ordinal permutation
            pattern = tuple(sorted(range(m_dim), key=lambda idx: sub_window[idx]))
            if pattern in pattern_counts:
                pattern_counts[pattern] += 1
            else:
                pattern_counts[pattern] = 1

        entropy = 0.0
        for count in pattern_counts.values():
            p = count / total_patterns
            entropy -= p * math.log(p)

        max_entropy = math.log(float(math.factorial(m_dim)))
        if max_entropy <= cfg.division_eps:
            return 0.0

        norm_pe = entropy / max_entropy
        return min(1.0, max(0.0, norm_pe))

    def compute_tsallis_entropy(
        self,
        returns: list[float],
        cfg: RegimeConfig,
    ) -> float:
        """Compute Tsallis Non-Extensive Entropy with power-law index q."""
        n_samples = len(returns)
        if n_samples == 0:
            raise InsufficientDataException("Returns list must not be empty.")

        q_idx = cfg.tsallis_q
        if abs(q_idx - 1.0) <= cfg.division_eps:
            return self.compute_shannon_entropy(returns, cfg)

        k_bins = cfg.entropy_bins
        min_val = min(returns)
        max_val = max(returns)
        val_range = max_val - min_val

        if val_range <= cfg.division_eps:
            return 0.0

        bin_width = val_range / k_bins
        bin_counts = [0] * k_bins

        for r in returns:
            idx = int((r - min_val) / bin_width)
            if idx >= k_bins:
                idx = k_bins - 1
            bin_counts[idx] += 1

        sum_pq = 0.0
        for count in bin_counts:
            if count > 0:
                p = count / n_samples
                sum_pq += p**q_idx

        tsallis = (1.0 - sum_pq) / (q_idx - 1.0)
        return max(0.0, tsallis)

    def classify_regime(
        self,
        hurst: float,
        yz_vol: float,
        has_jump: bool,
        cfg: RegimeConfig,
    ) -> MarketRegime:
        """Classify macroeconomic market regime deterministically."""
        if has_jump:
            return MarketRegime.HIGH_VOL_LIQUIDATION_CASCADE
        if hurst < 0.45:
            return MarketRegime.MEAN_REVERTING_RANGE
        if hurst > 0.55:
            return MarketRegime.STRONG_TREND_BULL
        return MarketRegime.RANDOM_WALK_NOISE
