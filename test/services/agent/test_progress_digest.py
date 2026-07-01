from types import SimpleNamespace

from app.services.agent.progress_digest import build_evidence_items, build_tool_result_digest
from app.services.source_evidence_ledger import stable_web_evidence_id


def test_web_search_digest_uses_tool_level_title_instead_of_first_source_title():
    handler = SimpleNamespace(
        _build_result_summary=lambda _result: {
            "kind": "search",
            "title": "OpenAI承诺在2026年对与AI相关的非营利问题投资5000万美元。",
            "count": 6,
            "truncated": False,
        }
    )
    record = SimpleNamespace(
        tool_call={"id": "call-1", "name": "web_search"},
        tool_name="web_search",
        result=SimpleNamespace(
            status="success",
            data={
                "sources": [
                    {
                        "title": "OpenAI承诺在2026年对与AI相关的非营利问题投资5000万美元。",
                        "url": "https://163.com/news",
                        "description": "第一条来源摘要",
                    },
                    {
                        "title": "OpenAI与博通联合发布首款自研AI推理芯片",
                        "url": "https://example.com/chip",
                        "description": "第二条来源摘要",
                    },
                ]
            },
            error_message=None,
        ),
        handler=handler,
    )

    digest = build_tool_result_digest(record)

    assert digest["title"] == "搜索完成"
    assert digest["summary"] == "保留 6 条候选结果，供后续回答筛选。"
    assert digest["source_refs"] == [
        stable_web_evidence_id("https://163.com/news", fallback="ev-call-1-0"),
        stable_web_evidence_id("https://example.com/chip", fallback="ev-call-1-1"),
    ]


def test_url_read_degraded_digest_does_not_expose_internal_service_names():
    handler = SimpleNamespace(
        _build_result_summary=lambda _result: {
            "kind": "url_read",
            "truncated": False,
        }
    )
    record = SimpleNamespace(
        tool_call={"id": "call-2", "name": "url_read"},
        tool_name="url_read",
        result=SimpleNamespace(
            status="degraded",
            data={"url": "https://example.com/a"},
            error_message="reader-service 返回 HTTP 502，已降级跳过",
        ),
        handler=handler,
    )

    digest = build_tool_result_digest(record)

    assert digest["title"] == "网页读取部分可用"
    assert digest["summary"] == "网页暂时无法读取，已跳过该来源。"
    assert digest["source_refs"] == [stable_web_evidence_id("https://example.com/a", fallback="ev-call-2-url")]
    assert "url_read" not in digest["title"]
    assert "reader-service" not in digest["summary"]


def test_url_read_success_builds_read_success_evidence():
    record = SimpleNamespace(
        tool_call={"id": "call-3", "name": "url_read"},
        tool_name="url_read",
        result=SimpleNamespace(
            status="success",
            data={
                "url": "https://www.example.com/report?utm_source=feed",
                "title": "Example Report",
                "content": "报告正文",
            },
            error_message=None,
        ),
        handler=None,
    )

    evidence = build_evidence_items(record)

    assert evidence == [
        {
            "id": stable_web_evidence_id("https://example.com/report", fallback="ev-call-3-url"),
            "kind": "web",
            "status": "read_success",
            "title": "Example Report",
            "url": "https://example.com/report",
            "domain": "example.com",
            "claim": "报告正文",
            "snippet": "报告正文",
            "used_by_final_answer": False,
        }
    ]


def test_duplicate_skipped_search_does_not_create_evidence_items():
    record = SimpleNamespace(
        tool_call={"id": "call-4", "name": "web_search"},
        tool_name="web_search",
        result=SimpleNamespace(
            status="degraded",
            data={
                "query": "OpenAI 最新公告 2026年6月 新闻",
                "sources": [],
                "duplicate_search_skipped": True,
            },
            error_message="重复搜索已跳过",
        ),
        handler=None,
    )

    evidence = build_evidence_items(record)

    assert evidence == []
