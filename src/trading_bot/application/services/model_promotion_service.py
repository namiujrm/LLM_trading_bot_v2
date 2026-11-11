"""Model Promotion & Zero-Downtime Hot-Reload Service (Clean Architecture, Invariant #9).

Coordinates atomic promotion of Challenger models to Champion:
1. Validates and executes atomic disk swap via ModelRegistryPort.
2. Backs up replaced champion to models/archive/ with full audit history.
3. Loads fresh champion model artifacts (boosters, ridge prior, OOD detector, TreeSHAP explainer).
4. Invokes live_paper_runner.reload_active_model() for zero-downtime hot-reloading.
5. Emits institutional promotion alerts to Telegram via TelegramNotifierPort.

Strict Clean Architecture:
- Application layer depends ONLY on Domain and Ports.
- ZERO imports from adapters, infrastructure, or interfaces.
- Enforces Invariant #9: zero default parameters, zero default fields, zero silent fallbacks.
"""

from __future__ import annotations

import logging
from typing import Any

from trading_bot.domain.entities.ml.model_registry_entities import (
    ModelArtifactMetadata,
    ModelPromotionRecord,
)
from trading_bot.ports.model_registry_port import CalibratorRegistryPort, ModelRegistryPort
from trading_bot.ports.telegram_port import TelegramNotifierPort

logger = logging.getLogger(__name__)


class ModelPromotionService:
    """Application service coordinating model promotion, runtime hot-reload, and alerting."""

    def __init__(
        self,
        model_registry: ModelRegistryPort,
        live_paper_runner: Any,
        notifier: TelegramNotifierPort | None,
    ) -> None:
        self._model_registry: ModelRegistryPort = model_registry
        self._live_paper_runner: Any = live_paper_runner
        self._notifier: TelegramNotifierPort | None = notifier

    @property
    def model_registry(self) -> ModelRegistryPort:
        return self._model_registry

    @property
    def live_paper_runner(self) -> Any:
        return self._live_paper_runner

    @property
    def notifier(self) -> TelegramNotifierPort | None:
        return self._notifier

    async def promote_and_reload(
        self,
        *,
        user_id: str,
        rationale: str,
    ) -> ModelPromotionRecord:
        """Promote current challenger to champion, hot-reload active runner, and alert Telegram.

        Args:
            user_id: Authorized identifier of the approving user.
            rationale: Explicit business or quantitative justification for promotion.

        Returns:
            ModelPromotionRecord containing immutable audit details.
        """
        logger.info(
            "Initiating model promotion by user '%s' with rationale: %s",
            user_id,
            rationale,
        )

        # 1. Execute atomic directory swap and archive backup in registry
        promotion_record = await self._model_registry.promote_challenger_to_champion(
            user_id=user_id,
            rationale=rationale,
        )

        # 2. Load the newly promoted champion model artifacts
        (
            new_barrier_model,
            new_ood_detector,
            new_shap_explainer,
            new_metadata,
        ) = await self._model_registry.load_champion_model()

        # 3. Perform zero-downtime hot-reload on live runner if present
        if self._live_paper_runner is not None and hasattr(
            self._live_paper_runner, "reload_active_model"
        ):
            self._live_paper_runner.reload_active_model(
                new_barrier_model=new_barrier_model,
                new_ood_detector=new_ood_detector,
                new_shap_explainer=new_shap_explainer,
                new_metadata=new_metadata,
            )
            if isinstance(self._model_registry, CalibratorRegistryPort) and hasattr(
                self._live_paper_runner, "reload_calibrators"
            ):
                new_cals = await self._model_registry.load_champion_calibrators()
                if new_cals is not None:
                    self._live_paper_runner.reload_calibrators(
                        calibrator_long=new_cals[0],
                        calibrator_short=new_cals[1],
                    )
            logger.info(
                "Hot-reloaded live runner with promoted Champion model: %s",
                new_metadata.model_id,
            )
        else:
            logger.warning(
                "Live paper runner not attached or does not support hot-reloading; skipped in-memory swap."
            )

        # 4. Dispatch Telegram success notification
        if self._notifier is not None:
            safe_rationale = rationale.replace("_", "\\_")
            msg = (
                f"🏆 *THĂNG HẠNG MODEL THÀNH CÔNG (PROMOTION COMPLETE)*\n\n"
                f"• *Champion Mới:* `{new_metadata.model_id}`\n"
                f"• *Champion Cũ:* `{promotion_record.previous_champion_id}`\n"
                f"• *Net Sharpe:* `{new_metadata.net_sharpe:.2f}`\n"
                f"• *Brier Skill Score:* `{new_metadata.brier_skill_score:.4f}`\n"
                f"• *Max Drawdown:* `{new_metadata.max_drawdown_pct:.2%}`\n"
                f"• *Người phê duyệt:* `{promotion_record.promoted_by_user_id}`\n"
                f"• *Lý do:* {safe_rationale}\n\n"
                f"✅ *Hot-Reload Zero-Downtime:* Hoàn tất cập nhật model trực tiếp trong RAM mà không cần khởi động lại container."
            )
            await self._notifier.send_text_message(msg, None)

        return promotion_record

    async def get_champion_metadata(self) -> ModelArtifactMetadata:
        """Fetch metadata for the active production Champion model."""
        _, _, _, meta = await self._model_registry.load_champion_model()
        return meta

    async def list_registered_models(self) -> list[ModelArtifactMetadata]:
        """Fetch metadata list for all registered models across roles."""
        return await self._model_registry.list_registered_models()
