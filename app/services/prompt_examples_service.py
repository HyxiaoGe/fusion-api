"""
示例问题服务

负责刷新、缓存、读取动态示例问题。
- 写入 PostgreSQL（持久化）
- 缓存到 Redis（TTL 2 小时）
- 读取时优先 Redis，miss 则查 PostgreSQL 并回填
"""
import json
import random
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logger import app_logger as logger
from app.core.redis import get_redis_pool
from app.db.database import SessionLocal
from app.db.models import PromptExample
from app.services.kimi_search_service import fetch_trending_questions

REDIS_KEY = "prompt:examples"
REDIS_TTL = 7200  # 2 小时

# 默认 fallback 问题（冷启动或 Kimi 不可用时使用）
DEFAULT_EXAMPLES = [
    {"category": "general", "question": "写一个 Python 快速排序函数"},
    {"category": "general", "question": "帮我 review 这段代码"},
    {"category": "tech", "question": "如何用 Docker 部署 FastAPI 服务"},
    {"category": "tech", "question": "解释 React useEffect 的执行时机"},
    {"category": "general", "question": "写一篇关于 AI 发展的短文"},
    {"category": "general", "question": "帮我润色这段产品介绍"},
    {"category": "general", "question": "写一封正式的商务邮件"},
    {"category": "general", "question": "解释一下量子计算的基本原理"},
]


async def refresh_prompt_examples() -> None:
    """
    定时任务入口：调用 Kimi 生成新问题 → 写 PostgreSQL → 写 Redis。
    """
    logger.info("开始刷新示例问题...")

    questions = await fetch_trending_questions()
    if not questions:
        logger.warning("未获取到新问题，跳过刷新")
        return

    db = SessionLocal()
    try:
        now = datetime.now(timezone(timedelta(hours=8)))
        expires_at = now + timedelta(hours=2)

        # 旧的标记为不活跃
        db.query(PromptExample).filter(PromptExample.is_active == True).update(
            {"is_active": False}
        )

        # 写入新的一批
        for item in questions:
            db.add(PromptExample(
                question=item["question"],
                category=item["category"],
                source="kimi",
                is_active=True,
                created_at=now,
                expires_at=expires_at,
            ))

        db.commit()
        logger.info(f"示例问题已写入数据库: {len(questions)} 条")

        # 写 Redis 缓存
        await _cache_to_redis(questions, now)

    except Exception as e:
        logger.error(f"刷新示例问题失败: {e}")
        db.rollback()
    finally:
        db.close()


async def get_prompt_examples(limit: int = 8) -> dict:
    """
    读取示例问题，优先 Redis，miss 则查 PostgreSQL。
    按 category 均匀采样后随机返回 limit 条。
    """
    # 1. 尝试 Redis
    cached = await _read_from_redis()
    if cached:
        examples = cached["examples"]
        refreshed_at = cached["refreshed_at"]
    else:
        # 2. 查 PostgreSQL
        examples, refreshed_at = _read_from_db()
        if examples:
            await _cache_to_redis(examples, refreshed_at)

    # 3. fallback
    if not examples:
        examples = DEFAULT_EXAMPLES
        refreshed_at = None

    # 4. 按 category 均匀采样
    sampled = _balanced_sample(examples, limit)

    return {
        "examples": sampled,
        "refreshed_at": refreshed_at.isoformat() if refreshed_at else None,
    }


def _balanced_sample(examples: list[dict], limit: int) -> list[dict]:
    """按 category 均匀采样，然后随机排序"""
    by_category: dict[str, list[dict]] = {}
    for item in examples:
        by_category.setdefault(item["category"], []).append(item)

    result = []
    categories = list(by_category.keys())
    if not categories:
        return []

    # 轮流从每个 category 取
    per_category = max(1, limit // len(categories))
    for cat in categories:
        items = by_category[cat]
        random.shuffle(items)
        result.extend(items[:per_category])

    # 补足到 limit
    remaining = [item for item in examples if item not in result]
    random.shuffle(remaining)
    result.extend(remaining[:limit - len(result)])

    random.shuffle(result)
    return result[:limit]


def _read_from_db() -> tuple[Optional[list[dict]], Optional[datetime]]:
    """从 PostgreSQL 读取活跃的示例问题"""
    db = SessionLocal()
    try:
        rows = db.query(PromptExample).filter(
            PromptExample.is_active == True
        ).order_by(PromptExample.created_at.desc()).all()

        if not rows:
            return None, None

        examples = [{"question": r.question, "category": r.category} for r in rows]
        refreshed_at = rows[0].created_at
        return examples, refreshed_at
    except Exception as e:
        logger.error(f"读取示例问题失败: {e}")
        return None, None
    finally:
        db.close()


async def _cache_to_redis(examples: list[dict], refreshed_at) -> None:
    """写入 Redis 缓存"""
    redis = get_redis_pool()
    if not redis:
        return
    try:
        data = {
            "examples": examples,
            "refreshed_at": refreshed_at.isoformat() if refreshed_at else None,
        }
        await redis.set(REDIS_KEY, json.dumps(data, ensure_ascii=False), ex=REDIS_TTL)
    except Exception as e:
        logger.warning(f"写入 Redis 缓存失败: {e}")


async def _read_from_redis() -> Optional[dict]:
    """从 Redis 读取缓存"""
    redis = get_redis_pool()
    if not redis:
        return None
    try:
        raw = await redis.get(REDIS_KEY)
        if raw:
            return json.loads(raw)
    except Exception as e:
        logger.warning(f"读取 Redis 缓存失败: {e}")
    return None
