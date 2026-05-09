"""自动探活 offline provider — Phase 2 cron job。

按 offline_reason 分档周期：
- key_invalid / quota_exceeded / other：30 min（admin 充值/换 key 后想立刻恢复）
- tos_blocked：24h（平台政策类，短期不会变，避免烧 token）

每 30 min 调度一次（scheduler_service 注册），job 内部根据 last_probe_at + interval 跳过未到期的。
探活成功 → 自动恢复（manual_recover by_user_id="auto_probe"），失败 → 仅更新 last_probe_at 等下次。
"""

from datetime import datetime, timedelta, timezone

from app.core.logger import app_logger as logger
from app.db.database import SessionLocal
from app.db.models import ModelSource, Provider
from app.services.health.provider_health import ProviderHealthService

CHINA_TZ = timezone(timedelta(hours=8))

# offline_reason → 探活间隔（分钟）
PROBE_INTERVALS_MINUTES = {
    "key_invalid": 30,
    "quota_exceeded": 30,
    "tos_blocked": 1440,  # 24h
    "other": 30,
}
DEFAULT_PROBE_INTERVAL_MINUTES = 30


async def probe_offline_providers() -> None:
    """轮询所有 offline provider，按 reason 决定是否到期 → 探活 → 成功则自动恢复"""
    from app.ai.llm_manager import llm_manager

    db = SessionLocal()
    try:
        offline_providers = db.query(Provider).filter(Provider.status == "offline").all()
        if not offline_providers:
            return

        now = datetime.now(CHINA_TZ).replace(tzinfo=None)
        logger.debug(f"自动探活：发现 {len(offline_providers)} 个 offline provider")

        for provider in offline_providers:
            interval_min = PROBE_INTERVALS_MINUTES.get(
                provider.offline_reason or "other", DEFAULT_PROBE_INTERVAL_MINUTES
            )
            # 用 last_probe_at 优先，没有则用 last_failure_at（首次探活前没探过）
            last_check = provider.last_probe_at or provider.last_failure_at
            if last_check and (now - last_check).total_seconds() < interval_min * 60:
                continue

            # 找该 provider 下任一启用的轻量模型
            model = (
                db.query(ModelSource)
                .filter(ModelSource.provider == provider.id, ModelSource.enabled)
                .order_by(ModelSource.priority.asc().nulls_last())
                .first()
            )
            if not model:
                logger.debug(f"自动探活：{provider.id} 无可用模型，跳过")
                continue

            # 探活：用系统 key（不传 user credentials），1 token "hi"
            try:
                result = await llm_manager.test_credentials(
                    provider=provider.id,
                    model_id=model.model_id,
                    credentials=None,
                    db=db,
                )
            except Exception as e:
                logger.warning(f"自动探活异常 [{provider.id}]: {e}")
                provider.last_probe_at = now
                db.commit()
                continue

            if result.get("valid"):
                ProviderHealthService(db).manual_recover(provider.id, by_user_id="auto_probe")
                logger.info(f"自动探活恢复：{provider.id}（之前 reason={provider.offline_reason}）")
            else:
                provider.last_probe_at = now
                db.commit()
                logger.debug(f"自动探活仍失败：{provider.id} reason={result.get('reason')} interval={interval_min}min")
    finally:
        db.close()
