"""Unit tests for Math Alpha Agent Adapter (Clean Architecture, Invariant #9).

Verifies:
1. Candidate generation across all 4 specialized causal clusters.
2. Robust JSON parsing (markdown code fences, <think> blocks, leading/trailing whitespace).
3. AST syntax validation (accepting valid expressions, rejecting invalid operators and negative lags).
4. Monotonic prior validation ({-1, 1} enforcement).
5. Error handling on malformed JSON, empty completions, missing required fields, and invalid clusters.
6. Target count limits and parameter validation.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from trading_bot.adapters.agents.math_alpha_agent import (
    CLUSTER_0_REGIME,
    CLUSTER_1_MICRO,
    CLUSTER_2_DERIV,
    CLUSTER_3_CROSS,
    MathAlphaAgentAdapter,
)
from trading_bot.adapters.agents.ninerouter_client import (
    NinerouterClient,
    NinerouterResponse,
)
from trading_bot.domain.entities.quant.symbolic_alpha import SymbolicAlphaCandidate
from trading_bot.domain.exceptions import (
    InvalidCausalClusterException,
    MalformedModelResponseException,
)
from trading_bot.domain.services.quant.alpha_ast_validator import AlphaASTValidator
from trading_bot.ports.rate_limit_governor_port import (
    RateLimitGovernorPort,
    RouteQuotaSpec,
)


class DummyRateLimitGovernor:
    """Deterministic dummy rate limit governor conforming to RateLimitGovernorPort."""

    async def acquire_available_route(self, tier: str, estimated_tokens: int) -> RouteQuotaSpec:
        return RouteQuotaSpec(
            route_id="dummy_route_01",
            model_name="deepseek-ai/deepseek-v4-pro",
            rpm_limit=60,
            tpm_limit=60000,
            rpd_limit=1000,
            min_interval_ms=100,
            max_concurrency=4,
        )

    async def mark_route_cooldown(self, route_id: str, cooldown_seconds: float) -> None:
        pass


def create_test_adapter(mock_client: NinerouterClient) -> MathAlphaAgentAdapter:
    """Helper to instantiate MathAlphaAgentAdapter with real AST validator and dummy governor."""
    governor: RateLimitGovernorPort = DummyRateLimitGovernor()
    ast_validator = AlphaASTValidator()
    return MathAlphaAgentAdapter(
        ninerouter_client=mock_client,
        rate_limit_governor=governor,
        model_tier="ds/deepseek-v4-pro",
        temperature=0.3,
        max_tokens=4096,
        ast_validator=ast_validator,
    )


def create_ninerouter_response(content: str, is_fallback: bool = False) -> NinerouterResponse:
    """Create a NinerouterResponse fixture."""
    return NinerouterResponse(
        content=content,
        model="deepseek-ai/deepseek-v4-pro",
        finish_reason="stop",
        prompt_tokens=150,
        completion_tokens=250,
        is_fallback=is_fallback,
    )


# ---------------------------------------------------------------------------
# 1. Candidate Generation Across All 4 Specialized Causal Clusters
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_candidates_cluster_0_regime() -> None:
    payload = [
        {
            "alpha_id": "wq_regime_hurst_01",
            "dsl_expression": "tanh_norm(ts_zscore(hurst, 20) - ts_zscore(yang_zhang_vol, 20))",
            "causal_cluster": CLUSTER_0_REGIME,
            "long_monotonic_prior": 1,
            "short_monotonic_prior": -1,
            "targeted_blindspot": "Regime switching transition lag",
            "economic_rationale": "High Hurst with low Yang-Zhang vol indicates persistent low-noise trending state",
        }
    ]
    mock_client = AsyncMock(spec=NinerouterClient)
    mock_client.generate_chat_completion.return_value = create_ninerouter_response(
        json.dumps(payload)
    )

    adapter = create_test_adapter(mock_client)
    candidates = await adapter.generate_candidates_for_cluster(
        cluster_name=CLUSTER_0_REGIME,
        blindspots=[{"issue": "trend_missed"}],
        decayed_alphas=["old_hurst_01"],
        target_count=5,
    )

    assert len(candidates) == 1
    c = candidates[0]
    assert isinstance(c, SymbolicAlphaCandidate)
    assert c.alpha_id == "wq_regime_hurst_01"
    assert c.causal_cluster == CLUSTER_0_REGIME
    assert c.long_monotonic_prior == 1
    assert c.short_monotonic_prior == -1
    assert "hurst" in c.dsl_expression


@pytest.mark.asyncio
async def test_generate_candidates_cluster_1_micro() -> None:
    payload = [
        {
            "alpha_id": "wq_micro_exhaustion_01",
            "dsl_expression": "tanh_norm(ts_zscore(ofi_sum, 16) - ts_zscore(vpin, 16))",
            "causal_cluster": CLUSTER_1_MICRO,
            "long_monotonic_prior": 1,
            "short_monotonic_prior": -1,
            "targeted_blindspot": "Toxic order flow absorption",
            "economic_rationale": "High OFI net buying with suppressed VPIN indicates informed accumulation",
        }
    ]
    mock_client = AsyncMock(spec=NinerouterClient)
    mock_client.generate_chat_completion.return_value = create_ninerouter_response(
        json.dumps(payload)
    )

    adapter = create_test_adapter(mock_client)
    candidates = await adapter.generate_candidates_for_cluster(
        cluster_name=CLUSTER_1_MICRO,
        blindspots=[],
        decayed_alphas=[],
        target_count=3,
    )

    assert len(candidates) == 1
    c = candidates[0]
    assert c.alpha_id == "wq_micro_exhaustion_01"
    assert c.causal_cluster == CLUSTER_1_MICRO
    assert "vpin" in c.dsl_expression


@pytest.mark.asyncio
async def test_generate_candidates_cluster_2_deriv() -> None:
    payload = [
        {
            "alpha_id": "wq_deriv_basis_01",
            "dsl_expression": "sign(ts_delta(funding_rate, 48)) * ts_zscore(basis_bps, 48)",
            "causal_cluster": CLUSTER_2_DERIV,
            "long_monotonic_prior": -1,
            "short_monotonic_prior": 1,
            "targeted_blindspot": "Perpetual funding rate squeeze trap",
            "economic_rationale": "Surging positive funding rate with premium basis flags crowded long leverage exhaustion",
        }
    ]
    mock_client = AsyncMock(spec=NinerouterClient)
    mock_client.generate_chat_completion.return_value = create_ninerouter_response(
        json.dumps(payload)
    )

    adapter = create_test_adapter(mock_client)
    candidates = await adapter.generate_candidates_for_cluster(
        cluster_name=CLUSTER_2_DERIV,
        blindspots=[],
        decayed_alphas=[],
        target_count=2,
    )

    assert len(candidates) == 1
    c = candidates[0]
    assert c.alpha_id == "wq_deriv_basis_01"
    assert c.causal_cluster == CLUSTER_2_DERIV
    assert c.long_monotonic_prior == -1
    assert c.short_monotonic_prior == 1


@pytest.mark.asyncio
async def test_generate_candidates_cluster_3_cross() -> None:
    payload = [
        {
            "alpha_id": "wq_cross_leadlag_01",
            "dsl_expression": "ts_zscore(btc_returns, 10) - ts_zscore(returns, 10)",
            "causal_cluster": CLUSTER_3_CROSS,
            "long_monotonic_prior": 1,
            "short_monotonic_prior": -1,
            "targeted_blindspot": "BTC leadership catchup delay",
            "economic_rationale": "Altcoin returns lagging BTC impulse indicates imminent beta catchup expansion",
        }
    ]
    mock_client = AsyncMock(spec=NinerouterClient)
    mock_client.generate_chat_completion.return_value = create_ninerouter_response(
        json.dumps(payload)
    )

    adapter = create_test_adapter(mock_client)
    candidates = await adapter.generate_candidates_for_cluster(
        cluster_name=CLUSTER_3_CROSS,
        blindspots=[],
        decayed_alphas=[],
        target_count=5,
    )

    assert len(candidates) == 1
    c = candidates[0]
    assert c.alpha_id == "wq_cross_leadlag_01"
    assert c.causal_cluster == CLUSTER_3_CROSS


@pytest.mark.asyncio
async def test_invalid_causal_cluster_raises_domain_exception() -> None:
    mock_client = AsyncMock(spec=NinerouterClient)
    adapter = create_test_adapter(mock_client)

    with pytest.raises(InvalidCausalClusterException, match="Unsupported causal cluster"):
        await adapter.generate_candidates_for_cluster(
            cluster_name="CLUSTER_99_UNKNOWN",
            blindspots=[],
            decayed_alphas=[],
            target_count=5,
        )


# ---------------------------------------------------------------------------
# 2. Robust JSON Cleanup (Markdown Fences, Think Blocks, Whitespace)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_json_cleanup_markdown_fences_with_json_tag() -> None:
    raw_response = """
    ```json
    [
      {
        "alpha_id": "wq_fenced_01",
        "dsl_expression": "tanh_norm(ts_zscore(returns, 16))",
        "causal_cluster": "CLUSTER_1_MICRO",
        "long_monotonic_prior": 1,
        "short_monotonic_prior": -1,
        "targeted_blindspot": "Oversold return bounce",
        "economic_rationale": "Normalized return momentum"
      }
    ]
    ```
    """
    mock_client = AsyncMock(spec=NinerouterClient)
    mock_client.generate_chat_completion.return_value = create_ninerouter_response(raw_response)

    adapter = create_test_adapter(mock_client)
    candidates = await adapter.generate_candidates_for_cluster(
        cluster_name=CLUSTER_1_MICRO,
        blindspots=[],
        decayed_alphas=[],
        target_count=1,
    )
    assert len(candidates) == 1
    assert candidates[0].alpha_id == "wq_fenced_01"


@pytest.mark.asyncio
async def test_json_cleanup_raw_fences_and_think_blocks() -> None:
    raw_response = """
    <think>
    Let's derive an order flow imbalance formula.
    We will subtract VPIN from OFI z-score.
    </think>
    ```
    [
      {
        "alpha_id": "wq_think_01",
        "dsl_expression": "tanh_norm(ts_zscore(ofi_sum, 16) - ts_zscore(vpin, 16))",
        "causal_cluster": "CLUSTER_1_MICRO",
        "long_monotonic_prior": 1,
        "short_monotonic_prior": -1,
        "targeted_blindspot": "Toxic divergence",
        "economic_rationale": "Institutional buying"
      }
    ]
    ```
    """
    mock_client = AsyncMock(spec=NinerouterClient)
    mock_client.generate_chat_completion.return_value = create_ninerouter_response(raw_response)

    adapter = create_test_adapter(mock_client)
    candidates = await adapter.generate_candidates_for_cluster(
        cluster_name=CLUSTER_1_MICRO,
        blindspots=[],
        decayed_alphas=[],
        target_count=1,
    )
    assert len(candidates) == 1
    assert candidates[0].alpha_id == "wq_think_01"


@pytest.mark.asyncio
async def test_json_wrapped_in_candidates_key_dict() -> None:
    raw_response = {
        "candidates": [
            {
                "alpha_id": "wq_dict_wrap_01",
                "dsl_expression": "ts_zscore(returns, 24)",
                "causal_cluster": CLUSTER_0_REGIME,
                "long_monotonic_prior": 1,
                "short_monotonic_prior": -1,
                "targeted_blindspot": "Trend momentum",
                "economic_rationale": "Standardized momentum",
            }
        ]
    }
    mock_client = AsyncMock(spec=NinerouterClient)
    mock_client.generate_chat_completion.return_value = create_ninerouter_response(
        json.dumps(raw_response)
    )

    adapter = create_test_adapter(mock_client)
    candidates = await adapter.generate_candidates_for_cluster(
        cluster_name=CLUSTER_0_REGIME,
        blindspots=[],
        decayed_alphas=[],
        target_count=1,
    )
    assert len(candidates) == 1
    assert candidates[0].alpha_id == "wq_dict_wrap_01"


# ---------------------------------------------------------------------------
# 3. AST Syntax Validation & Operator Whitelist
# ---------------------------------------------------------------------------


def test_ast_validation_accepts_valid_expressions() -> None:
    mock_client = AsyncMock(spec=NinerouterClient)
    adapter = create_test_adapter(mock_client)

    valid_expressions = [
        "tanh_norm(ts_zscore(ofi_sum, 16) - ts_zscore(vpin, 16))",
        "sign(ts_delta(hurst, 5)) * ts_zscore(returns, 20)",
        "decay_linear(ts_rank(open_interest, 24), 12)",
        "ts_corr(returns, vwap, 48)",
        "signed_power(ts_zscore(close, 20), 2.0)",
        "ts_zscore(btc_returns, 10) - ts_zscore(returns, 10)",
        "ts_min(low, 24) / ts_max(high, 24)",
        "log(abs(volume) + 1.0)",
    ]

    for expr in valid_expressions:
        assert adapter.validate_expression(expr) is True, f"Failed for valid expr: {expr}"


def test_ast_validation_rejects_invalid_operators_and_lookahead() -> None:
    mock_client = AsyncMock(spec=NinerouterClient)
    adapter = create_test_adapter(mock_client)

    invalid_expressions = [
        "eval('1 + 1')",
        "__import__('os').system('ls')",
        "unsupported_func(close, 10)",
        "open + high + unknown_identifier_col",
        "ts_delay(close, -5)",  # Negative look-ahead lag
        "ts_delay(close, 0)",  # Zero lag (must be >= 1)
        "ts_mean(close, -1)",  # Negative window
        "",  # Empty
        "   ",  # Whitespace
        "def bad(): pass",  # Statement instead of expression
    ]

    for expr in invalid_expressions:
        assert adapter.validate_expression(expr) is False, f"Expected reject for: {expr}"


@pytest.mark.asyncio
async def test_candidate_with_invalid_ast_is_rejected_from_generation() -> None:
    payload = [
        {
            "alpha_id": "wq_valid_01",
            "dsl_expression": "tanh_norm(ts_zscore(ofi_sum, 16) - ts_zscore(vpin, 16))",
            "causal_cluster": CLUSTER_1_MICRO,
            "long_monotonic_prior": 1,
            "short_monotonic_prior": -1,
            "targeted_blindspot": "Valid formula",
            "economic_rationale": "Valid mechanism",
        },
        {
            "alpha_id": "wq_malicious_02",
            "dsl_expression": 'eval(\'__import__("os").system("whoami")\')',
            "causal_cluster": CLUSTER_1_MICRO,
            "long_monotonic_prior": 1,
            "short_monotonic_prior": -1,
            "targeted_blindspot": "Exploit attempt",
            "economic_rationale": "Malicious code injection",
        },
    ]
    mock_client = AsyncMock(spec=NinerouterClient)
    mock_client.generate_chat_completion.return_value = create_ninerouter_response(
        json.dumps(payload)
    )

    adapter = create_test_adapter(mock_client)
    candidates = await adapter.generate_candidates_for_cluster(
        cluster_name=CLUSTER_1_MICRO,
        blindspots=[],
        decayed_alphas=[],
        target_count=5,
    )

    # Malicious candidate must be discarded by AST gatekeeper
    assert len(candidates) == 1
    assert candidates[0].alpha_id == "wq_valid_01"


# ---------------------------------------------------------------------------
# 4. Monotonic Prior Validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_monotonic_prior_invalid_long_prior_raises() -> None:
    payload = [
        {
            "alpha_id": "wq_bad_prior_01",
            "dsl_expression": "ts_zscore(returns, 10)",
            "causal_cluster": CLUSTER_1_MICRO,
            "long_monotonic_prior": 0,  # Invalid: must be in {-1, 1}
            "short_monotonic_prior": -1,
            "targeted_blindspot": "Invalid prior",
            "economic_rationale": "Test prior constraint",
        }
    ]
    mock_client = AsyncMock(spec=NinerouterClient)
    mock_client.generate_chat_completion.return_value = create_ninerouter_response(
        json.dumps(payload)
    )

    adapter = create_test_adapter(mock_client)
    with pytest.raises(
        MalformedModelResponseException, match=r"Invalid long_monotonic_prior=0\. Must be -1 or 1\."
    ):
        await adapter.generate_candidates_for_cluster(
            cluster_name=CLUSTER_1_MICRO,
            blindspots=[],
            decayed_alphas=[],
            target_count=1,
        )


@pytest.mark.asyncio
async def test_monotonic_prior_invalid_short_prior_raises() -> None:
    payload = [
        {
            "alpha_id": "wq_bad_prior_02",
            "dsl_expression": "ts_zscore(returns, 10)",
            "causal_cluster": CLUSTER_1_MICRO,
            "long_monotonic_prior": 1,
            "short_monotonic_prior": 2,  # Invalid: must be in {-1, 1}
            "targeted_blindspot": "Invalid prior",
            "economic_rationale": "Test prior constraint",
        }
    ]
    mock_client = AsyncMock(spec=NinerouterClient)
    mock_client.generate_chat_completion.return_value = create_ninerouter_response(
        json.dumps(payload)
    )

    adapter = create_test_adapter(mock_client)
    with pytest.raises(
        MalformedModelResponseException,
        match=r"Invalid short_monotonic_prior=2\. Must be -1 or 1\.",
    ):
        await adapter.generate_candidates_for_cluster(
            cluster_name=CLUSTER_1_MICRO,
            blindspots=[],
            decayed_alphas=[],
            target_count=1,
        )


# ---------------------------------------------------------------------------
# 5. Error Handling: Malformed JSON, Empty Completions, Missing Fields
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_completion_raises_malformed_model_response() -> None:
    mock_client = AsyncMock(spec=NinerouterClient)
    mock_client.generate_chat_completion.return_value = create_ninerouter_response("")

    adapter = create_test_adapter(mock_client)
    with pytest.raises(MalformedModelResponseException, match="Empty completion"):
        await adapter.generate_candidates_for_cluster(
            cluster_name=CLUSTER_0_REGIME,
            blindspots=[],
            decayed_alphas=[],
            target_count=3,
        )


@pytest.mark.asyncio
async def test_whitespace_completion_raises_malformed_model_response() -> None:
    mock_client = AsyncMock(spec=NinerouterClient)
    mock_client.generate_chat_completion.return_value = create_ninerouter_response("   \n\t  ")

    adapter = create_test_adapter(mock_client)
    with pytest.raises(MalformedModelResponseException, match="Empty completion"):
        await adapter.generate_candidates_for_cluster(
            cluster_name=CLUSTER_0_REGIME,
            blindspots=[],
            decayed_alphas=[],
            target_count=3,
        )


@pytest.mark.asyncio
async def test_fallback_response_raises_malformed_model_response() -> None:
    mock_client = AsyncMock(spec=NinerouterClient)
    mock_client.generate_chat_completion.return_value = create_ninerouter_response(
        "", is_fallback=True
    )

    adapter = create_test_adapter(mock_client)
    with pytest.raises(
        MalformedModelResponseException, match="Empty completion or 9Router fallback"
    ):
        await adapter.generate_candidates_for_cluster(
            cluster_name=CLUSTER_0_REGIME,
            blindspots=[],
            decayed_alphas=[],
            target_count=3,
        )


@pytest.mark.asyncio
async def test_unparseable_json_raises_malformed_model_response() -> None:
    mock_client = AsyncMock(spec=NinerouterClient)
    mock_client.generate_chat_completion.return_value = create_ninerouter_response(
        "Here are your alphas: [broken json: {alpha_id: ..."
    )

    adapter = create_test_adapter(mock_client)
    with pytest.raises(MalformedModelResponseException, match="Could not parse valid JSON"):
        await adapter.generate_candidates_for_cluster(
            cluster_name=CLUSTER_0_REGIME,
            blindspots=[],
            decayed_alphas=[],
            target_count=3,
        )


@pytest.mark.asyncio
async def test_missing_required_field_raises_malformed_model_response() -> None:
    # Missing 'economic_rationale'
    payload = [
        {
            "alpha_id": "wq_missing_field_01",
            "dsl_expression": "ts_zscore(returns, 10)",
            "causal_cluster": CLUSTER_1_MICRO,
            "long_monotonic_prior": 1,
            "short_monotonic_prior": -1,
            "targeted_blindspot": "Missing rationale",
        }
    ]
    mock_client = AsyncMock(spec=NinerouterClient)
    mock_client.generate_chat_completion.return_value = create_ninerouter_response(
        json.dumps(payload)
    )

    adapter = create_test_adapter(mock_client)
    with pytest.raises(
        MalformedModelResponseException, match="missing required field 'economic_rationale'"
    ):
        await adapter.generate_candidates_for_cluster(
            cluster_name=CLUSTER_1_MICRO,
            blindspots=[],
            decayed_alphas=[],
            target_count=3,
        )


# ---------------------------------------------------------------------------
# 6. Target Count Slicing and Validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_target_count_slicing() -> None:
    payload = [
        {
            "alpha_id": f"wq_slice_{i}",
            "dsl_expression": f"ts_zscore(returns, {10 + i})",
            "causal_cluster": CLUSTER_0_REGIME,
            "long_monotonic_prior": 1,
            "short_monotonic_prior": -1,
            "targeted_blindspot": "Batch",
            "economic_rationale": "Batch rationale",
        }
        for i in range(5)
    ]
    mock_client = AsyncMock(spec=NinerouterClient)
    mock_client.generate_chat_completion.return_value = create_ninerouter_response(
        json.dumps(payload)
    )

    adapter = create_test_adapter(mock_client)
    candidates = await adapter.generate_candidates_for_cluster(
        cluster_name=CLUSTER_0_REGIME,
        blindspots=[],
        decayed_alphas=[],
        target_count=2,
    )

    assert len(candidates) == 2
    assert candidates[0].alpha_id == "wq_slice_0"
    assert candidates[1].alpha_id == "wq_slice_1"


@pytest.mark.asyncio
async def test_invalid_target_count_raises_value_error() -> None:
    mock_client = AsyncMock(spec=NinerouterClient)
    adapter = create_test_adapter(mock_client)

    with pytest.raises(ValueError, match="target_count must be positive"):
        await adapter.generate_candidates_for_cluster(
            cluster_name=CLUSTER_0_REGIME,
            blindspots=[],
            decayed_alphas=[],
            target_count=0,
        )
