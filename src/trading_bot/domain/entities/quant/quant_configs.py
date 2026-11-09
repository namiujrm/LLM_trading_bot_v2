"""Quantitative Engine Configuration Schemas — Pure Domain Models (Zero Defaults).

Invariant #9: All fields are strictly required with zero defaults, zero class-field
assignments, and zero silent fallbacks.
"""

from __future__ import annotations

from pydantic import BaseModel


class VPINConfig(BaseModel):
    """Configuration for Volume-Synchronized Probability of Informed Trading and Microstructure."""

    num_buckets: int
    adv_lookback_days: int
    volume_bucket_slice_ratio: float
    toxicity_high_threshold: float
    division_eps: float


class RegimeConfig(BaseModel):
    """Configuration for Statistical Physics, Volatility, and Regime Detection."""

    hurst_window: int
    yang_zhang_window: int
    bipower_jump_confidence: float
    entropy_bins: int
    permutation_dimension: int
    tsallis_q: float
    division_eps: float


class FracDiffConfig(BaseModel):
    """Configuration for Fixed-Window Fractional Differentiation."""

    d_order: float
    weight_threshold: float
    max_window_length: int
    adf_critical_pvalue: float


class AlphaDSLConfig(BaseModel):
    """Configuration for Symbolic Alpha DSL AST Compiler and Validator."""

    max_ast_depth: int
    max_ast_nodes: int
    min_window_bars: int
    max_window_bars: int
    division_epsilon: float


class ICGatekeeperConfig(BaseModel):
    """Configuration for Rolling Spearman Rank Information Coefficient (IC) Gatekeeper."""

    min_rank_ic: float
    min_t_statistic: float
    min_ic_ir: float
    rolling_eval_window_bars: int
    forward_prediction_bars: int


class QuantEngineMasterConfig(BaseModel):
    """Master configuration aggregating all quant engine sub-module configurations."""

    vpin: VPINConfig
    regime: RegimeConfig
    fracdiff: FracDiffConfig
    alpha_dsl: AlphaDSLConfig
    ic_gatekeeper: ICGatekeeperConfig
