"""Math Alpha Agent Adapter (Clean Architecture, Invariant #9).

Generates symbolic alpha candidates using DeepSeek Pro via 9Router gateway
across 4 specialized causal clusters:
- CLUSTER_0_REGIME: Market regime switching, Hurst exponent, Yang-Zhang vol, FracDiff trend.
- CLUSTER_1_MICRO: Microstructure exhaustion, OFI (Order Flow Imbalance), VPIN, OBI.
- CLUSTER_2_DERIV: Derivatives & OI, funding rate basis, liquidation cascades.
- CLUSTER_3_CROSS: Cross-asset lead-lag, BTC lead-lag cascade on ETH/SOL/BNB, beta divergence.

Enforces Invariant #9 (Zero Defaults, Zero Class Field Defaults, Zero Silent Fallbacks).
AST Syntax and monotonic prior validation before candidate acceptance.
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
from typing import Any, cast

from trading_bot.adapters.agents.ninerouter_client import (
    NinerouterClient,
    NinerouterResponse,
)
from trading_bot.domain.entities.quant.quant_configs import AlphaDSLConfig
from trading_bot.domain.entities.quant.symbolic_alpha import SymbolicAlphaCandidate
from trading_bot.domain.exceptions import (
    InvalidCausalClusterException,
    MalformedModelResponseException,
)
from trading_bot.domain.services.quant.alpha_ast_validator import AlphaASTValidator
from trading_bot.ports.math_alpha_agent_port import MathAlphaAgentPort
from trading_bot.ports.rate_limit_governor_port import RateLimitGovernorPort

_std_logger: logging.Logger = logging.getLogger(__name__)

CLUSTER_0_REGIME: str = "CLUSTER_0_REGIME"
CLUSTER_1_MICRO: str = "CLUSTER_1_MICRO"
CLUSTER_2_DERIV: str = "CLUSTER_2_DERIV"
CLUSTER_3_CROSS: str = "CLUSTER_3_CROSS"

SUPPORTED_CLUSTERS: frozenset[str] = frozenset(
    {
        CLUSTER_0_REGIME,
        CLUSTER_1_MICRO,
        CLUSTER_2_DERIV,
        CLUSTER_3_CROSS,
    }
)

CLUSTER_DOMAIN_PROMPTS: dict[str, str] = {
    CLUSTER_0_REGIME: (
        "Specialized Causal Cluster: CLUSTER_0_REGIME (Market Regime Switching & Statistical Physics).\n"
        "Domain Mechanism: Financial markets undergo non-linear phase transitions between persistent trending, "
        "mean-reverting consolidation, and volatility regime shifts. "
        "Allowed domain features: 'hurst' (Anis-Lloyd / Peng Hurst exponent), 'yang_zhang_vol' (overnight jump + open-to-close volatility), "
        "'fracdiff_close' (fractionally differentiated stationary close price), 'returns', 'close', 'volume', 'spread_bps_mean'. "
        "Formulate expressions capturing regime transitions, volatility jumps, and long-memory stationarity."
    ),
    CLUSTER_1_MICRO: (
        "Specialized Causal Cluster: CLUSTER_1_MICRO (Microstructure Exhaustion & Order Flow Dynamics).\n"
        "Domain Mechanism: High-frequency market microstructure exhibits toxic order flow, aggressive taker imbalances, "
        "and bid-ask inventory depletion prior to sharp short-term reversals or trend exhaustion. "
        "Allowed domain features: 'ofi_sum' (Order Flow Imbalance multi-level cumulative sum), 'vpin' (Volume-Synchronized Probability of Toxicity), "
        "'obi' (Order Book Imbalance ratio), 'spread_bps_mean' (effective bid-ask spread), 'volume', 'taker_buy_vol'. "
        "Formulate expressions detecting adverse selection, liquidity drain, and toxic flow imbalances."
    ),
    CLUSTER_2_DERIV: (
        "Specialized Causal Cluster: CLUSTER_2_DERIV (Derivatives, Open Interest & Basis Dynamics).\n"
        "Domain Mechanism: Crypto perpetual futures markets are driven by leverage positioning, funding rate basis divergence, "
        "and liquidation cascades. Crowded directional positioning with extreme funding rates leads to violent squeeze cascades. "
        "Allowed domain features: 'funding_rate' (8h perpetual funding rate basis), 'open_interest' (contract open interest), "
        "'basis_bps' (perpetual mark price vs spot index price spread), 'liquidation_vol' (forced liquidation volume), 'volume', 'returns'. "
        "Formulate expressions exploiting crowded leverage positioning, basis mean-reversion, and liquidation pressure."
    ),
    CLUSTER_3_CROSS: (
        "Specialized Causal Cluster: CLUSTER_3_CROSS (Cross-Asset Lead-Lag & Beta Divergence).\n"
        "Domain Mechanism: Bitcoin (BTC) is the structural liquidity anchor and market leader in cryptocurrency markets. "
        "Major altcoins (ETH, SOL, BNB) exhibit systematic lead-lag delays, beta overextension, and catch-up momentum cascades. "
        "Allowed domain features: 'btc_returns' (BTC benchmark returns), 'beta_divergence' (rolling CAPM beta divergence from equilibrium), "
        "'returns', 'vwap', 'volume', 'close'. "
        "Formulate expressions capturing BTC spillover delays, beta catch-up momentum, and cross-asset relative value divergences."
    ),
}


class MathAlphaAgentAdapter(MathAlphaAgentPort):
    """Concrete adapter for Math Alpha Agent communicating via 9Router with AST validation."""

    def __init__(
        self,
        ninerouter_client: NinerouterClient,
        rate_limit_governor: RateLimitGovernorPort,
        model_tier: str,
        temperature: float,
        max_tokens: int,
        ast_validator: AlphaASTValidator,
    ) -> None:
        self._client: NinerouterClient = ninerouter_client
        self._governor: RateLimitGovernorPort = rate_limit_governor
        self._model_tier: str = model_tier
        self._temperature: float = temperature
        self._max_tokens: int = max_tokens
        self._ast_validator: AlphaASTValidator = ast_validator

    def _build_cluster_prompt(
        self,
        cluster_name: str,
        blindspots: list[dict[str, Any]],
        decayed_alphas: list[str],
        target_count: int,
    ) -> list[dict[str, str]]:
        cluster_domain_desc: str = CLUSTER_DOMAIN_PROMPTS[cluster_name]

        system_prompt: str = (
            "You are the Math Alpha Quantitative Evolution Agent in an institutional crypto hedge fund.\n"
            "Your role is to discover robust, economically justified symbolic alpha expressions "
            "expressed in valid Symbolic Polars AST format.\n\n"
            f"{cluster_domain_desc}\n\n"
            "Mathematical Operator Whitelist:\n"
            "- Unary transforms: abs(x), sign(x), log(x), sqrt(x), tanh_norm(x)\n"
            "- Rolling time-series operators: ts_delay(x, d), ts_delta(x, d), ts_mean(x, d), "
            "ts_std(x, d), ts_zscore(x, d), ts_rank(x, d), ts_min(x, d), ts_max(x, d), decay_linear(x, d)\n"
            "- Binary and interaction operators: +, -, *, /, signed_power(x, a) [with a in (0, 10]], ts_corr(x, y, d)\n"
            "- STRICT CAUSAL INVARIANT: Every window parameter 'd' MUST be a strictly positive integer >= 1 (e.g. 5, 10, 16, 20, 48, 96). "
            "Negative lags and look-ahead references are strictly forbidden.\n\n"
            "You MUST output ONLY a valid JSON array of objects conforming to the following schema:\n"
            "[\n"
            "  {\n"
            '    "alpha_id": string,\n'
            '    "dsl_expression": string,\n'
            f'    "causal_cluster": "{cluster_name}",\n'
            '    "long_monotonic_prior": 1 or -1,\n'
            '    "short_monotonic_prior": 1 or -1,\n'
            '    "targeted_blindspot": string,\n'
            '    "economic_rationale": string\n'
            "  }\n"
            "]\n"
            "Do NOT output markdown commentary or text outside the JSON array or code block.\n"
            "IMPORTANT: Keep your internal reasoning concise (under 250 tokens). Immediately formulate and output the final JSON array."
        )

        user_content_dict: dict[str, Any] = {
            "cluster_name": cluster_name,
            "target_candidate_count": target_count,
            "recent_blindspots": blindspots,
            "decayed_alphas_to_replace": decayed_alphas,
        }

        user_prompt: str = json.dumps(user_content_dict, indent=2)

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _extract_raw_candidates(self, raw_text: str) -> list[dict[str, Any]]:
        clean_text: str = re.sub(r"<think>[\s\S]*?</think>", "", raw_text).strip()
        if len(clean_text) == 0:
            raise MalformedModelResponseException(
                "Model output contains only empty or think blocks."
            )

        fence_pattern: str = r"```(?:json)?\s*([\s\S]*?)\s*```"
        match = re.search(fence_pattern, clean_text)
        payload_text = match.group(1).strip() if match is not None else clean_text

        parsed: object | None = None
        try:
            parsed = json.loads(payload_text)
        except Exception:
            start_arr = payload_text.find("[")
            end_arr = payload_text.rfind("]")
            if start_arr != -1 and end_arr != -1 and end_arr > start_arr:
                with contextlib.suppress(Exception):
                    parsed = json.loads(payload_text[start_arr : end_arr + 1])

            if parsed is None:
                start_obj = payload_text.find("{")
                end_obj = payload_text.rfind("}")
                if start_obj != -1 and end_obj != -1 and end_obj > start_obj:
                    with contextlib.suppress(Exception):
                        parsed = json.loads(payload_text[start_obj : end_obj + 1])

        if parsed is None:
            raise MalformedModelResponseException(
                f"Could not parse valid JSON from model completion: {raw_text[:160]}"
            )

        if isinstance(parsed, list):
            items: list[dict[str, Any]] = []
            for elem in cast(list[object], parsed):
                if isinstance(elem, dict):
                    items.append(cast(dict[str, Any], elem))
            return items

        if isinstance(parsed, dict):
            for key in ("candidates", "alphas", "symbolic_alphas", "formulas"):
                if key in parsed and isinstance(parsed[key], list):
                    items_dict: list[dict[str, Any]] = []
                    for elem in cast(list[object], parsed[key]):
                        if isinstance(elem, dict):
                            items_dict.append(cast(dict[str, Any], elem))
                    return items_dict
            if "alpha_id" in parsed and "dsl_expression" in parsed:
                return [cast(dict[str, Any], parsed)]
            raise MalformedModelResponseException(
                "Expected JSON list of alpha candidates, got dict."
            )

        type_name: str = type(parsed).__name__
        raise MalformedModelResponseException(
            f"Expected JSON list of alpha candidates, got: {type_name}"
        )

    def validate_expression(self, expression: str) -> bool:
        """Validate whether an expression conforms to AST security and syntax rules."""
        expr_clean: str = expression.strip()
        if not expr_clean:
            return False

        try:
            cfg = AlphaDSLConfig(
                max_ast_depth=10,
                max_ast_nodes=64,
                min_window_bars=1,
                max_window_bars=500,
                division_epsilon=1e-8,
            )
            self._ast_validator.validate_formula(formula=expr_clean, cfg=cfg)
            return True
        except Exception:
            return False

    def _validate_and_parse_candidate(
        self,
        item: dict[str, Any],
        cluster_name: str,
    ) -> SymbolicAlphaCandidate | None:
        required_fields = (
            "alpha_id",
            "dsl_expression",
            "causal_cluster",
            "long_monotonic_prior",
            "short_monotonic_prior",
            "targeted_blindspot",
            "economic_rationale",
        )
        for field in required_fields:
            if field not in item or item[field] is None:
                raise MalformedModelResponseException(
                    f"Candidate payload missing required field '{field}': {item}"
                )

        raw_long = item["long_monotonic_prior"]
        raw_short = item["short_monotonic_prior"]

        if not isinstance(raw_long, int) or isinstance(raw_long, bool) or raw_long not in (-1, 1):
            raise MalformedModelResponseException(
                f"Invalid long_monotonic_prior={raw_long}. Must be -1 or 1."
            )

        if (
            not isinstance(raw_short, int)
            or isinstance(raw_short, bool)
            or raw_short not in (-1, 1)
        ):
            raise MalformedModelResponseException(
                f"Invalid short_monotonic_prior={raw_short}. Must be -1 or 1."
            )

        dsl_expr: str = str(item["dsl_expression"]).strip()
        if not self.validate_expression(dsl_expr):
            _std_logger.warning("Rejected alpha candidate with invalid AST syntax: %s", dsl_expr)
            return None

        alpha_id: str = str(item["alpha_id"]).strip()
        target_cluster: str = str(item["causal_cluster"]).strip()
        blindspot: str = str(item["targeted_blindspot"]).strip()
        rationale: str = str(item["economic_rationale"]).strip()

        return SymbolicAlphaCandidate(
            alpha_id=alpha_id,
            dsl_expression=dsl_expr,
            causal_cluster=target_cluster,
            long_monotonic_prior=raw_long,
            short_monotonic_prior=raw_short,
            targeted_blindspot=blindspot,
            economic_rationale=rationale,
        )

    async def generate_candidates_for_cluster(
        self,
        cluster_name: str,
        blindspots: list[dict[str, Any]],
        decayed_alphas: list[str],
        target_count: int,
    ) -> list[SymbolicAlphaCandidate]:
        """Generate symbolic alpha candidates targeted at a specific causal cluster."""
        if cluster_name not in SUPPORTED_CLUSTERS:
            raise InvalidCausalClusterException(
                f"Unsupported causal cluster '{cluster_name}'. Supported clusters: {sorted(SUPPORTED_CLUSTERS)}"
            )
        if target_count <= 0:
            raise ValueError(f"target_count must be positive, got {target_count}")

        messages = self._build_cluster_prompt(
            cluster_name=cluster_name,
            blindspots=blindspots,
            decayed_alphas=decayed_alphas,
            target_count=target_count,
        )

        response: NinerouterResponse = await self._client.generate_chat_completion(
            tier=self._model_tier,
            messages=messages,
            estimated_tokens=2500,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )

        if response.is_fallback or not response.content or len(response.content.strip()) == 0:
            raise MalformedModelResponseException("Empty completion or 9Router fallback received.")

        raw_candidates = self._extract_raw_candidates(response.content)
        valid_candidates: list[SymbolicAlphaCandidate] = []

        for item in raw_candidates:
            candidate = self._validate_and_parse_candidate(item, cluster_name)
            if candidate is not None:
                valid_candidates.append(candidate)

        return valid_candidates[:target_count]
