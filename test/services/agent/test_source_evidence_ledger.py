from app.services.source_evidence_ledger import (
    build_search_source_evidence_item,
    build_url_read_evidence_item,
    stable_web_evidence_id,
)


def test_search_and_url_read_share_stable_evidence_id():
    search_item = build_search_source_evidence_item(
        {
            "title": "Example 官方公告",
            "url": "https://www.Example.com/news/launch?utm_source=newsletter&b=2&a=1#section",
            "description": "搜索摘要",
        },
        tool_call_id="tc-search",
        source_index=0,
    )
    read_item = build_url_read_evidence_item(
        {
            "url": "https://example.com/news/launch?a=1&b=2",
            "title": "Example 官方公告",
            "content": "网页正文",
        },
        status="success",
        tool_call_id="tc-read",
    )

    assert search_item["id"] == stable_web_evidence_id(
        "https://example.com/news/launch?b=2&a=1",
        fallback="ev-tc-search-0",
    )
    assert search_item["id"] == read_item["id"]
    assert search_item["url"] == "https://example.com/news/launch?a=1&b=2"


def test_search_candidate_uses_candidate_status():
    item = build_search_source_evidence_item(
        {
            "title": "OpenAI 官方公告",
            "url": "https://openai.com/index/example",
            "description": "OpenAI 发布官方公告。",
        },
        tool_call_id="tc-search",
        source_index=1,
    )

    assert item["status"] == "candidate"
    assert item["kind"] == "web"
    assert item["domain"] == "openai.com"
    assert item["claim"] == "OpenAI 发布官方公告。"


def test_url_read_status_maps_to_read_lifecycle():
    success_item = build_url_read_evidence_item(
        {
            "url": "https://example.com/success",
            "title": "成功页面",
            "content": "正文内容",
        },
        status="success",
        tool_call_id="tc-read-success",
    )
    degraded_item = build_url_read_evidence_item(
        {
            "url": "https://example.com/degraded",
            "title": "降级页面",
            "reason": "HTTP 502",
        },
        status="degraded",
        tool_call_id="tc-read-degraded",
    )
    failed_item = build_url_read_evidence_item(
        {
            "url": "https://example.com/failed",
            "reason": "timeout",
        },
        status="failed",
        tool_call_id="tc-read-failed",
    )

    assert success_item is not None
    assert degraded_item is not None
    assert failed_item is not None
    assert success_item["status"] == "read_success"
    assert degraded_item["status"] == "read_degraded"
    assert failed_item["status"] == "read_failed"
