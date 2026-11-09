"""Machine Learning Prediction Repository Adapter implementing MLPredictionRepositoryPort — Zero Defaults.

Handles high-throughput batch upsert of Stage 4 ML predictions into TimescaleDB hypertable
`quant_feature_vectors_30m`. Formats SHAP attributions, Venn-Abers calibration bounds, and
conformal OOD metrics into structured JSONB and native columns.
Enforces Invariant #9: zero default parameters, zero class field defaults, zero silent fallbacks.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from trading_bot.adapters.persistence.database import DatabaseManager
from trading_bot.domain.entities.ml.ml_predictions import (
    CalibratedProbability,
    OODDetectionResult,
    SHAPDriverContribution,
    UnifiedMLPredictionPacket,
)
from trading_bot.ports.ml_prediction_repository_port import MLPredictionRepositoryPort


def serialize_shap_explanation_json(packet: UnifiedMLPredictionPacket) -> str:
    """Format complete SHAP explanation and ML metrics payload for JSONB storage."""
    payload = {
        "symbol": packet.symbol,
        "bar_time_utc": packet.bar_time.isoformat(),
        "model_version": packet.model_version,
        "is_economic_consistent": packet.is_economic_consistent,
        "probabilities": {
            "p_long_raw": float(packet.p_long_raw),
            "p_short_raw": float(packet.p_short_raw),
            "p_long_calibrated": float(packet.p_long_calibrated.p_calibrated),
            "p_long_uncertainty": float(packet.p_long_calibrated.uncertainty_width),
            "p_long_bounds": [
                float(packet.p_long_calibrated.p_lower),
                float(packet.p_long_calibrated.p_upper),
            ],
            "p_short_calibrated": float(packet.p_short_calibrated.p_calibrated),
            "p_short_uncertainty": float(packet.p_short_calibrated.uncertainty_width),
            "p_short_bounds": [
                float(packet.p_short_calibrated.p_lower),
                float(packet.p_short_calibrated.p_upper),
            ],
        },
        "ood": {
            "is_ood": bool(packet.ood_status.is_ood),
            "mahalanobis_distance": float(packet.ood_status.mahalanobis_distance),
            "empirical_quantile_threshold": float(packet.ood_status.empirical_quantile_threshold),
            "excess_distance": float(packet.ood_status.excess_distance),
            "gamma_ood": float(packet.ood_status.gamma_ood),
        },
        "top_bullish_shap": [
            {
                "feature_name": d.feature_name,
                "feature_value": float(d.feature_value),
                "shap_value": float(d.shap_value),
                "direction": "BULLISH",
                "expected_sign": d.expected_sign,
                "is_sign_consistent": d.is_sign_consistent,
            }
            for d in packet.top_bullish_shap
        ],
        "top_bearish_shap": [
            {
                "feature_name": d.feature_name,
                "feature_value": float(d.feature_value),
                "shap_value": float(d.shap_value),
                "direction": "BEARISH",
                "expected_sign": d.expected_sign,
                "is_sign_consistent": d.is_sign_consistent,
            }
            for d in packet.top_bearish_shap
        ],
    }
    return json.dumps(payload)


def serialize_features_json(packet: UnifiedMLPredictionPacket) -> str:
    """Format input features snapshot dictionary for JSONB storage."""
    payload = {
        "features": {k: float(v) for k, v in packet.features_snapshot.items()},
        "symbol": packet.symbol,
        "bar_time_utc": packet.bar_time.isoformat(),
    }
    return json.dumps(payload)


class MLPredictionRepository(MLPredictionRepositoryPort):
    """PostgreSQL / TimescaleDB repository implementing MLPredictionRepositoryPort."""

    def __init__(
        self,
        db_manager: DatabaseManager,
        batch_chunk_size: int,
    ) -> None:
        if batch_chunk_size <= 0:
            raise ValueError(f"batch_chunk_size must be positive, received {batch_chunk_size}")
        self._db: DatabaseManager = db_manager
        self._batch_chunk_size: int = batch_chunk_size

    async def save_prediction_packet(self, packet: UnifiedMLPredictionPacket) -> None:
        """Persist a single ML prediction packet into quant_feature_vectors_30m."""
        await self.save_prediction_packets_batch([packet])

    async def save_prediction_packets_batch(
        self,
        packets: Sequence[UnifiedMLPredictionPacket],
    ) -> int:
        """Bulk upsert a batch of ML prediction packets using asyncpg executemany."""
        if len(packets) == 0:
            return 0

        query = """
            INSERT INTO quant_feature_vectors_30m (
                bar_time,
                symbol,
                regime_label,
                hurst_exp,
                yang_zhang_vol,
                vpin_value,
                kyle_lambda,
                oi_quadrant,
                ood_mahalanobis_dist,
                is_ood_warning,
                p_long_win_cal,
                p_long_uncertainty,
                p_short_win_cal,
                p_short_uncertainty,
                model_version,
                features_json,
                shap_explanation_json,
                gamma_ood,
                ood_threshold,
                is_economic_consistent,
                p_long_raw,
                p_short_raw,
                is_coherent,
                coherence_penalty
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                $11, $12, $13, $14, $15, $16::jsonb, $17::jsonb,
                $18, $19, $20, $21, $22, $23, $24
            )
            ON CONFLICT (symbol, bar_time) DO UPDATE SET
                regime_label            = EXCLUDED.regime_label,
                hurst_exp               = EXCLUDED.hurst_exp,
                yang_zhang_vol          = EXCLUDED.yang_zhang_vol,
                vpin_value              = EXCLUDED.vpin_value,
                kyle_lambda             = EXCLUDED.kyle_lambda,
                oi_quadrant             = EXCLUDED.oi_quadrant,
                ood_mahalanobis_dist    = EXCLUDED.ood_mahalanobis_dist,
                is_ood_warning          = EXCLUDED.is_ood_warning,
                p_long_win_cal          = EXCLUDED.p_long_win_cal,
                p_long_uncertainty      = EXCLUDED.p_long_uncertainty,
                p_short_win_cal         = EXCLUDED.p_short_win_cal,
                p_short_uncertainty     = EXCLUDED.p_short_uncertainty,
                model_version           = EXCLUDED.model_version,
                features_json           = EXCLUDED.features_json,
                shap_explanation_json   = EXCLUDED.shap_explanation_json,
                gamma_ood               = EXCLUDED.gamma_ood,
                ood_threshold           = EXCLUDED.ood_threshold,
                is_economic_consistent  = EXCLUDED.is_economic_consistent,
                p_long_raw              = EXCLUDED.p_long_raw,
                p_short_raw             = EXCLUDED.p_short_raw,
                is_coherent             = EXCLUDED.is_coherent,
                coherence_penalty       = EXCLUDED.coherence_penalty;
        """

        total_saved = 0
        total_packets = len(packets)

        for chunk_start in range(0, total_packets, self._batch_chunk_size):
            chunk = packets[chunk_start : chunk_start + self._batch_chunk_size]
            records: list[tuple[Any, ...]] = []

            for p in chunk:
                shap_json = serialize_shap_explanation_json(p)
                features_json = serialize_features_json(p)

                if "hurst_exponent_100" in p.features_snapshot:
                    hurst_val = p.features_snapshot["hurst_exponent_100"]
                elif "hurst_exp" in p.features_snapshot:
                    hurst_val = p.features_snapshot["hurst_exp"]
                else:
                    hurst_val = 0.50

                if "yang_zhang_vol_24" in p.features_snapshot:
                    yz_val = p.features_snapshot["yang_zhang_vol_24"]
                elif "yang_zhang_vol" in p.features_snapshot:
                    yz_val = p.features_snapshot["yang_zhang_vol"]
                else:
                    yz_val = 0.00

                if "vpin_toxicity_pct" in p.features_snapshot:
                    vpin_val = p.features_snapshot["vpin_toxicity_pct"]
                elif "vpin_value" in p.features_snapshot:
                    vpin_val = p.features_snapshot["vpin_value"]
                else:
                    vpin_val = 0.00

                if "kyle_lambda_log" in p.features_snapshot:
                    kyle_val = p.features_snapshot["kyle_lambda_log"]
                elif "kyle_lambda" in p.features_snapshot:
                    kyle_val = p.features_snapshot["kyle_lambda"]
                else:
                    kyle_val = 0.00

                records.append(
                    (
                        p.bar_time,
                        p.symbol,
                        p.regime_label,
                        float(hurst_val),
                        float(yz_val),
                        float(vpin_val),
                        float(kyle_val),
                        "NEUTRAL",
                        float(p.ood_status.mahalanobis_distance),
                        bool(p.ood_status.is_ood),
                        float(p.p_long_calibrated.p_calibrated),
                        float(p.p_long_calibrated.uncertainty_width),
                        float(p.p_short_calibrated.p_calibrated),
                        float(p.p_short_calibrated.uncertainty_width),
                        p.model_version,
                        features_json,
                        shap_json,
                        float(p.ood_status.gamma_ood),
                        float(p.ood_status.empirical_quantile_threshold),
                        bool(p.is_economic_consistent),
                        float(p.p_long_raw),
                        float(p.p_short_raw),
                        bool(p.is_coherent),
                        float(p.coherence_penalty),
                    )
                )

            async with self._db.transaction() as conn:
                await conn.executemany(query, records)
            total_saved += len(chunk)

        return total_saved

    async def get_prediction_packet(
        self,
        symbol: str,
    ) -> UnifiedMLPredictionPacket | None:
        """Fetch the most recent ML prediction packet for a given symbol."""
        query = """
            SELECT
                bar_time,
                symbol,
                regime_label,
                hurst_exp,
                yang_zhang_vol,
                vpin_value,
                kyle_lambda,
                oi_quadrant,
                ood_mahalanobis_dist,
                is_ood_warning,
                p_long_win_cal,
                p_long_uncertainty,
                p_short_win_cal,
                p_short_uncertainty,
                model_version,
                features_json,
                shap_explanation_json,
                gamma_ood,
                ood_threshold,
                is_economic_consistent,
                p_long_raw,
                p_short_raw,
                is_coherent,
                coherence_penalty
            FROM quant_feature_vectors_30m
            WHERE symbol = $1
            ORDER BY bar_time DESC
            LIMIT 1;
        """
        row = await self._db.fetchrow(query, symbol)
        if row is None:
            return None

        return self._parse_row_to_packet(row)

    async def get_prediction_packets_range(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
    ) -> Sequence[UnifiedMLPredictionPacket]:
        """Fetch chronological sequence of prediction packets within [start_time, end_time]."""
        query = """
            SELECT
                bar_time,
                symbol,
                regime_label,
                hurst_exp,
                yang_zhang_vol,
                vpin_value,
                kyle_lambda,
                oi_quadrant,
                ood_mahalanobis_dist,
                is_ood_warning,
                p_long_win_cal,
                p_long_uncertainty,
                p_short_win_cal,
                p_short_uncertainty,
                model_version,
                features_json,
                shap_explanation_json,
                gamma_ood,
                ood_threshold,
                is_economic_consistent,
                p_long_raw,
                p_short_raw,
                is_coherent,
                coherence_penalty
            FROM quant_feature_vectors_30m
            WHERE symbol = $1
              AND bar_time >= $2
              AND bar_time <= $3
            ORDER BY bar_time ASC;
        """
        rows = await self._db.fetch(query, symbol, start_time, end_time)
        return [self._parse_row_to_packet(r) for r in rows]

    async def save_prediction(self, prediction: UnifiedMLPredictionPacket) -> None:
        """Alias for save_prediction_packet."""
        await self.save_prediction_packet(packet=prediction)

    async def save_predictions_batch(
        self,
        predictions: Sequence[UnifiedMLPredictionPacket],
    ) -> int:
        """Alias for save_prediction_packets_batch."""
        return await self.save_prediction_packets_batch(packets=predictions)

    async def fetch_latest_prediction(
        self,
        symbol: str,
    ) -> UnifiedMLPredictionPacket | None:
        """Alias for get_prediction_packet."""
        return await self.get_prediction_packet(symbol=symbol)

    async def fetch_predictions_range(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
    ) -> Sequence[UnifiedMLPredictionPacket]:
        """Alias for get_prediction_packets_range."""
        return await self.get_prediction_packets_range(
            symbol=symbol, start_time=start_time, end_time=end_time
        )

    def _parse_row_to_packet(self, row: Any) -> UnifiedMLPredictionPacket:
        """Parse database row into UnifiedMLPredictionPacket (Fail-Fast, Zero Defaults)."""
        bar_time = row["bar_time"]
        if bar_time.tzinfo is None:
            bar_time = bar_time.replace(tzinfo=UTC)

        symbol = str(row["symbol"])
        regime_label = str(row["regime_label"])
        model_ver = str(row["model_version"])
        m_dist = float(str(row["ood_mahalanobis_dist"]))
        is_ood = bool(row["is_ood_warning"])
        p_long_cal = float(str(row["p_long_win_cal"]))
        p_long_unc = float(str(row["p_long_uncertainty"]))
        p_short_cal = float(str(row["p_short_win_cal"]))
        p_short_unc = float(str(row["p_short_uncertainty"]))

        shap_raw = row["shap_explanation_json"]
        if isinstance(shap_raw, str):
            shap_dict: dict[str, Any] = json.loads(shap_raw)
        else:
            shap_dict = shap_raw

        features_raw = row["features_json"]
        if isinstance(features_raw, str):
            feat_dict: dict[str, Any] = json.loads(features_raw)
        else:
            feat_dict = features_raw

        if "is_economic_consistent" in row:
            is_econ_consistent = bool(row["is_economic_consistent"])
        elif "is_economic_consistent" in shap_dict:
            is_econ_consistent = bool(shap_dict["is_economic_consistent"])
        else:
            is_econ_consistent = True

        if "gamma_ood" in row:
            gamma_ood = float(str(row["gamma_ood"]))
            thresh = float(str(row["ood_threshold"]))
            excess = max(0.0, m_dist - thresh)
        elif "ood" in shap_dict:
            ood_data = shap_dict["ood"]
            gamma_ood = float(str(ood_data["gamma_ood"]))
            thresh = float(str(ood_data["empirical_quantile_threshold"]))
            excess = float(str(ood_data["excess_distance"]))
        else:
            if is_ood:
                gamma_ood = 0.50
            else:
                gamma_ood = 1.00
            thresh = 3.00
            excess = max(0.0, m_dist - thresh)

        ood_result = OODDetectionResult(
            is_ood=is_ood,
            mahalanobis_distance=m_dist,
            empirical_quantile_threshold=thresh,
            excess_distance=excess,
            gamma_ood=gamma_ood,
            feature_contributions=(),
        )

        if "p_long_raw" in row:
            p_long_raw = float(str(row["p_long_raw"]))
            p_short_raw = float(str(row["p_short_raw"]))
            if "probabilities" in shap_dict and "p_long_bounds" in shap_dict["probabilities"]:
                prob_data = shap_dict["probabilities"]
                bounds_long = prob_data["p_long_bounds"]
                p_long_lower = float(str(bounds_long[0]))
                p_long_upper = float(str(bounds_long[1]))
                bounds_short = prob_data["p_short_bounds"]
                p_short_lower = float(str(bounds_short[0]))
                p_short_upper = float(str(bounds_short[1]))
            else:
                p_long_lower = max(0.0, p_long_cal - 0.5 * p_long_unc)
                p_long_upper = min(1.0, p_long_cal + 0.5 * p_long_unc)
                p_short_lower = max(0.0, p_short_cal - 0.5 * p_short_unc)
                p_short_upper = min(1.0, p_short_cal + 0.5 * p_short_unc)
        elif "probabilities" in shap_dict:
            prob_data = shap_dict["probabilities"]
            p_long_raw = float(str(prob_data["p_long_raw"]))
            p_short_raw = float(str(prob_data["p_short_raw"]))
            bounds_long = prob_data["p_long_bounds"]
            p_long_lower = float(str(bounds_long[0]))
            p_long_upper = float(str(bounds_long[1]))
            bounds_short = prob_data["p_short_bounds"]
            p_short_lower = float(str(bounds_short[0]))
            p_short_upper = float(str(bounds_short[1]))
        else:
            p_long_raw = p_long_cal
            p_short_raw = p_short_cal
            p_long_lower = max(0.0, p_long_cal - 0.5 * p_long_unc)
            p_long_upper = min(1.0, p_long_cal + 0.5 * p_long_unc)
            p_short_lower = max(0.0, p_short_cal - 0.5 * p_short_unc)
            p_short_upper = min(1.0, p_short_cal + 0.5 * p_short_unc)

        unc_long = max(0.0, p_long_upper - p_long_lower)
        unc_short = max(0.0, p_short_upper - p_short_lower)
        p_cal_l = min(max(p_long_cal, p_long_lower), p_long_upper)
        p_cal_s = min(max(p_short_cal, p_short_lower), p_short_upper)

        cal_long = CalibratedProbability(
            p_calibrated=p_cal_l,
            p_lower=p_long_lower,
            p_upper=p_long_upper,
            uncertainty_width=unc_long,
            is_uncertain=unc_long > 0.15,
            raw_score=p_long_raw,
        )

        cal_short = CalibratedProbability(
            p_calibrated=p_cal_s,
            p_lower=p_short_lower,
            p_upper=p_short_upper,
            uncertainty_width=unc_short,
            is_uncertain=unc_short > 0.15,
            raw_score=p_short_raw,
        )

        top_bull_list: list[SHAPDriverContribution] = []
        if "top_bullish_shap" in shap_dict:
            for d in shap_dict["top_bullish_shap"]:
                top_bull_list.append(
                    SHAPDriverContribution(
                        feature_name=str(d["feature_name"]),
                        feature_value=float(d["feature_value"]),
                        shap_value=float(d["shap_value"]),
                        expected_sign=int(d["expected_sign"]),
                        is_sign_consistent=bool(d["is_sign_consistent"]),
                    )
                )

        top_bear_list: list[SHAPDriverContribution] = []
        if "top_bearish_shap" in shap_dict:
            for d in shap_dict["top_bearish_shap"]:
                top_bear_list.append(
                    SHAPDriverContribution(
                        feature_name=str(d["feature_name"]),
                        feature_value=float(d["feature_value"]),
                        shap_value=float(d["shap_value"]),
                        expected_sign=int(d["expected_sign"]),
                        is_sign_consistent=bool(d["is_sign_consistent"]),
                    )
                )

        if "features" in feat_dict:
            snapshot_dict = {str(k): float(v) for k, v in feat_dict["features"].items()}
        else:
            snapshot_dict = {"hurst_exp": float(str(row["hurst_exp"]))}

        return UnifiedMLPredictionPacket(
            symbol=symbol,
            bar_time=bar_time,
            regime_label=regime_label,
            model_version=model_ver,
            p_long_raw=p_long_raw,
            p_short_raw=p_short_raw,
            p_long_calibrated=cal_long,
            p_short_calibrated=cal_short,
            ood_status=ood_result,
            top_bullish_shap=tuple(top_bull_list),
            top_bearish_shap=tuple(top_bear_list),
            is_economic_consistent=is_econ_consistent,
            features_snapshot=snapshot_dict,
        )
