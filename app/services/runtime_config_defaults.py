"""运行时配置默认值。

这些默认值是 DB 配置不可用时的稳定 fallback，也是 Alembic seed 的来源说明。
"""

from __future__ import annotations

from collections.abc import Iterator
from uuid import NAMESPACE_URL, uuid5

from app.ai.prompts.agent_loop import (
    APP_IDENTITY_PROMPT,
    CONTINUATION_SYSTEM_PROMPT,
    LIMIT_SUMMARY_PROMPT,
    NO_TOOL_NETWORK_BOUNDARY_PROMPT,
    NO_VISION_FILE_BOUNDARY_PROMPT,
    TOOL_USAGE_CONTRACT_PROMPT,
    URL_READ_TOOL_DESCRIPTION,
)
from app.ai.prompts.templates import (
    FILE_ANALYSIS_PROMPT,
    FILE_CONTENT_ENHANCEMENT_PROMPT,
    GENERATE_SUGGESTED_QUESTIONS_PROMPT,
    GENERATE_TITLE_PROMPT,
)

DEFAULT_MODEL_PRESENTATION_CONFIG = {
    "long_context_threshold_tokens": 128000,
    "weights": {
        "base": 40,
        "network": 25,
        "vision": 15,
        "long_context": 15,
        "deep_thinking": 10,
    },
    "levels": {
        "recommended": 85,
        "capable": 70,
    },
    "copy": {
        "unavailable_headline": "不建议：当前不可用",
        "default_headline": "适合：稳定知识与普通对话",
        "network_headline": "推荐：实时资料与复杂查询",
        "network_vision_headline": "推荐：实时资料和图片理解",
        "network_long_context_headline": "推荐：实时资料与长任务",
        "network_vision_long_context_headline": "推荐：实时资料、图片和长任务",
        "vision_headline": "适合：图片理解与普通对话",
        "base_reason": "可处理普通文本任务",
        "network_reason": "可联网搜索并读取关键来源",
        "vision_reason": "支持图片理解",
        "long_context_reason": "适合长上下文任务",
        "deep_thinking_reason": "适合复杂推理",
        "no_network_warning": "不支持实时联网，涉及最新信息时会基于已有知识谨慎回答",
        "network_tooltip": "可按问题需要自主联网搜索和读取关键来源",
        "no_network_tooltip": "不支持联网搜索，将基于模型知识回答",
        "vision_tooltip": "支持读图和图片理解",
        "no_vision_tooltip": "不支持图片理解",
        "deep_thinking_tooltip": "适合复杂推理和深度任务",
        "no_deep_thinking_tooltip": "不支持深度思考模式",
        "unhealthy_fallback": "服务商暂时不可用",
    },
}


DEFAULT_PROMPT_TEMPLATES = {
    "app_identity": APP_IDENTITY_PROMPT,
    "tool_usage_contract": TOOL_USAGE_CONTRACT_PROMPT,
    "no_tool_network_boundary": NO_TOOL_NETWORK_BOUNDARY_PROMPT,
    "no_vision_file_boundary": NO_VISION_FILE_BOUNDARY_PROMPT,
    "url_read_tool_description": URL_READ_TOOL_DESCRIPTION,
    "limit_summary": LIMIT_SUMMARY_PROMPT,
    "continuation_system": CONTINUATION_SYSTEM_PROMPT,
    "generate_title": GENERATE_TITLE_PROMPT,
    "generate_suggested_questions": GENERATE_SUGGESTED_QUESTIONS_PROMPT,
    "file_analysis": FILE_ANALYSIS_PROMPT,
    "file_content_enhancement": FILE_CONTENT_ENHANCEMENT_PROMPT,
}


def iter_default_runtime_config_seed_rows() -> Iterator[dict]:
    """生成 runtime_config_entries v1 seed rows。"""

    yield {
        "id": _seed_id("agent_strategy", "default"),
        "namespace": "agent_strategy",
        "key": "default",
        "version": "2026-07-02.v1",
        "payload": DEFAULT_AGENT_STRATEGY_CONFIG,
        "is_active": True,
        "description": "Agent 搜索、深读、来源排序、工具上下文默认策略",
    }
    yield {
        "id": _seed_id("model_presentation", "default"),
        "namespace": "model_presentation",
        "key": "default",
        "version": "2026-07-02.v1",
        "payload": DEFAULT_MODEL_PRESENTATION_CONFIG,
        "is_active": True,
        "description": "模型能力展示默认文案和评分规则",
    }
    for key, template in DEFAULT_PROMPT_TEMPLATES.items():
        yield {
            "id": _seed_id("prompt_template", key),
            "namespace": "prompt_template",
            "key": key,
            "version": "2026-07-02.v1",
            "payload": {"template": template},
            "is_active": True,
            "description": f"Prompt 模板：{key}",
        }


def _seed_id(namespace: str, key: str, version: str = "2026-07-02.v1") -> str:
    return str(uuid5(NAMESPACE_URL, f"fusion/runtime-config/{namespace}/{key}/{version}"))


DEFAULT_AGENT_STRATEGY_CONFIG = {
    "model_runtime": {
        "agent_tools_disabled_aliases": ["qwen-vl-max"],
    },
    "search": {
        "standard_budget": {
            "name": "standard",
            "requested_count": 5,
            "context_source_limit": 5,
        },
        "budgets_by_intent": {
            "quick_fact": {"name": "quick_fact", "requested_count": 3, "context_source_limit": 3},
            "freshness": {"name": "freshness", "requested_count": 5, "context_source_limit": 5},
            "comparison": {"name": "comparison", "requested_count": 8, "context_source_limit": 6},
            "deep_research": {"name": "deep_research", "requested_count": 10, "context_source_limit": 8},
            "official_source": {"name": "official_source", "requested_count": 5, "context_source_limit": 4},
        },
        "followup_budgets_by_name": {
            "standard": {"name": "standard_followup", "requested_count": 3, "context_source_limit": 3},
            "quick_fact": {"name": "quick_fact_followup", "requested_count": 3, "context_source_limit": 3},
            "freshness": {"name": "freshness_followup", "requested_count": 3, "context_source_limit": 3},
            "official_source": {"name": "official_source_followup", "requested_count": 3, "context_source_limit": 3},
            "comparison": {"name": "comparison_followup", "requested_count": 5, "context_source_limit": 4},
            "deep_research": {"name": "deep_research_followup", "requested_count": 5, "context_source_limit": 5},
        },
        "intent_keywords": {
            "comparison": [
                "权威媒体",
                "媒体",
                "报道",
                "对照",
                "对比",
                "比较",
                "compare",
                "comparison",
                "versus",
                "media",
                "reuters",
                "bloomberg",
                "techcrunch",
                "axios",
                "bbc",
                "nytimes",
                "new york times",
                "wall street journal",
                "wsj",
                "the verge",
            ],
            "official_source": [
                "官方",
                "官网",
                "公告",
                "发布",
                "official",
                "announcement",
                "announces",
                "announced",
                "press release",
                "release notes",
                "official blog",
                "openai.com",
            ],
            "deep_research": [
                "深入",
                "调研",
                "研究",
                "论文",
                "白皮书",
                "技术报告",
                "technical report",
                "system card",
                "research",
                "paper",
                "whitepaper",
            ],
            "freshness": [
                "最新",
                "今天",
                "今日",
                "目前",
                "实时",
                "current",
                "latest",
                "today",
                "recent",
                "new",
            ],
            "quick_fact": [
                "是谁",
                "是什么",
                "多少",
                "价格",
                "上市日期",
                "who is",
                "what is",
                "when did",
                "how much",
            ],
        },
        "thresholds": {
            "similar_followup": 0.55,
            "duplicate_search": 0.82,
        },
    },
    "network": {
        "max_search_calls": 4,
        "default_planned_search_calls": 2,
        "deep_research_planned_search_calls": 3,
        "max_url_read_calls": 5,
        "max_domains": 5,
        "repair_search_count": 3,
        "repair_context_source_limit": 3,
        "weak_search_result_threshold": 2,
        "min_recency_days": 1,
        "max_recency_days": 365,
    },
    "read_planner": {
        "read_limits": {
            "quick_fact": 1,
            "standard": 2,
            "deep": 3,
        },
        "quick_budget_names": ["quick_fact", "quick_fact_followup"],
        "freshness_budget_names": ["freshness", "freshness_followup"],
        "deep_budget_names": [
            "comparison",
            "comparison_followup",
            "deep_research",
            "deep_research_followup",
            "official_source",
            "official_source_followup",
        ],
        "deep_intents": ["comparison", "deep_research", "official_source"],
    },
    "source_ranker": {
        "max_low_priority_examples": 3,
        "authority_media_domains": [
            "apnews.com",
            "axios.com",
            "bloomberg.com",
            "cnbc.com",
            "ft.com",
            "nytimes.com",
            "reuters.com",
            "techcrunch.com",
            "theverge.com",
            "venturebeat.com",
            "wired.com",
            "wsj.com",
        ],
        "low_priority_domains": [
            "bilibili.com",
            "douyin.com",
            "facebook.com",
            "instagram.com",
            "reddit.com",
            "threads.com",
            "tiktok.com",
            "twitter.com",
            "weibo.com",
            "x.com",
            "youtube.com",
            "youtu.be",
            "zhihu.com",
        ],
        "video_domains": ["bilibili.com", "douyin.com", "tiktok.com", "youtube.com", "youtu.be"],
        "forum_domains": ["reddit.com", "threads.com", "twitter.com", "weibo.com", "x.com", "zhihu.com"],
        "weights": {
            "source_order_base": 22,
            "source_order_step": 2,
            "official": 38,
            "original": 22,
            "specific_original": 18,
            "official_original": 35,
            "pdf": 35,
            "authority_media": 36,
            "listing_penalty": 28,
            "video_penalty": 28,
            "forum_penalty": 24,
            "low_priority_penalty": 18,
            "relevance_per_term": 4,
            "relevance_max": 18,
        },
        "priority_thresholds": {
            "high": 60,
            "medium": 30,
        },
    },
    "tool_context": {
        "max_context_sources": 8,
        "max_sources_per_domain": 2,
        "url_read_max_content_chars": 8000,
        "url_read_max_reason_chars": 160,
    },
}
