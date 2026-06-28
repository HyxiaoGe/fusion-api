from types import SimpleNamespace

from app.services.agent.progress_digest import build_tool_result_digest


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
    assert digest["source_refs"] == ["ev-call-1-0", "ev-call-1-1"]
