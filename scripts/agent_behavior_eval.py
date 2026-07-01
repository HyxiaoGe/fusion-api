"""
Agent 行为评估集 V1。

默认 dry-run 只输出样本基线，不调用 LLM、搜索服务或浏览器。
真实 Chrome 回归可以把观测结果转成 observation 后复用 score_observation()。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TextIO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SAMPLE_PATH = PROJECT_ROOT / "test" / "fixtures" / "agent_behavior_eval_samples.json"

REQUIRED_SAMPLE_FIELDS = {"id", "category", "question", "expected_tool_policy", "expected_surface"}
VALID_TOOL_POLICIES = {"no_search", "search"}
VALID_SURFACES = {"direct_answer", "evidence"}
SEARCH_SURFACES = {"execution_process", "answer_evidence"}
OPTIONAL_BOOL_FIELDS = {"requires_search_keywords", "requires_console_clean"}


def load_samples(path: Path = DEFAULT_SAMPLE_PATH) -> list[dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("评估样本必须是数组")

    seen_ids: set[str] = set()
    samples: list[dict] = []
    for index, item in enumerate(raw, 1):
        if not isinstance(item, dict):
            raise ValueError(f"第 {index} 个样本必须是对象")

        missing = sorted(REQUIRED_SAMPLE_FIELDS - set(item))
        if missing:
            raise ValueError(f"样本缺少字段: id={item.get('id')}, fields={missing}")

        sample_id = str(item["id"]).strip()
        if not sample_id:
            raise ValueError(f"第 {index} 个样本 id 为空")
        if sample_id in seen_ids:
            raise ValueError(f"重复样本 id: {sample_id}")
        seen_ids.add(sample_id)

        policy = item.get("expected_tool_policy")
        if policy not in VALID_TOOL_POLICIES:
            raise ValueError(f"expected_tool_policy 非法: id={sample_id}, value={policy}")

        surface = item.get("expected_surface")
        if surface not in VALID_SURFACES:
            raise ValueError(f"expected_surface 非法: id={sample_id}, value={surface}")

        if not str(item.get("category", "")).strip():
            raise ValueError(f"样本 category 为空: id={sample_id}")
        if not str(item.get("question", "")).strip():
            raise ValueError(f"样本 question 为空: id={sample_id}")

        for field in OPTIONAL_BOOL_FIELDS:
            if field in item and not isinstance(item[field], bool):
                raise ValueError(f"{field} 必须是布尔值: id={sample_id}")

        samples.append(item)

    return samples


def score_observation(sample: dict, observation: dict) -> dict:
    issues: list[str] = []
    if not observation:
        issues.append("缺少观测结果")

    tool_calls = _string_set(observation.get("tool_calls", []))
    surfaces = _string_set(observation.get("surfaces", []))
    output_text = _combined_output_text(observation)

    for error in _string_list(observation.get("console_errors", [])):
        issues.append(f"存在 console error: {error}")

    _check_tool_policy(sample, tool_calls, issues)
    _check_surface_policy(sample, surfaces, issues)
    _check_search_context(sample, observation, issues)
    _check_forbidden_terms(sample, observation, output_text, issues)

    return {
        "passed": not issues,
        "issues": issues,
        "tool_calls": sorted(tool_calls),
        "surfaces": sorted(surfaces),
    }


def write_dry_run(samples: list[dict], output: TextIO = sys.stdout) -> None:
    for sample in samples:
        score = score_observation(sample, {})
        output.write(
            json.dumps(
                {
                    "sample_id": sample["id"],
                    "category": sample["category"],
                    "question": sample["question"],
                    "expected_tool_policy": sample["expected_tool_policy"],
                    "expected_surface": sample["expected_surface"],
                    "passed": score["passed"],
                    "issues": score["issues"],
                },
                ensure_ascii=False,
            )
        )
        output.write("\n")


def _check_tool_policy(sample: dict, tool_calls: set[str], issues: list[str]) -> None:
    policy = sample.get("expected_tool_policy")
    if policy == "no_search":
        for tool_name in sorted(tool_calls & {"web_search", "url_read"}):
            issues.append(f"no_search 场景不应调用 {tool_name}")
    elif policy == "search" and "web_search" not in tool_calls:
        issues.append("search 场景必须调用 web_search")


def _check_surface_policy(sample: dict, surfaces: set[str], issues: list[str]) -> None:
    surface = sample.get("expected_surface")
    if surface == "direct_answer":
        for surface_name in sorted(surfaces & SEARCH_SURFACES):
            issues.append(f"direct_answer 场景不应展示 {surface_name}")
    elif surface == "evidence":
        for surface_name in sorted(SEARCH_SURFACES - surfaces):
            issues.append(f"evidence 场景应展示 {surface_name}")


def _check_search_context(sample: dict, observation: dict, issues: list[str]) -> None:
    if sample.get("requires_search_keywords") and not _string_list(observation.get("search_keywords", [])):
        issues.append("搜索场景应展示搜索关键词")

    min_sources = sample.get("min_sources")
    if isinstance(min_sources, int) and min_sources > 0:
        source_count = observation.get("source_count", 0)
        if not isinstance(source_count, int):
            source_count = 0
        if source_count < min_sources:
            issues.append(f"来源数量不足: actual={source_count} min={min_sources}")


def _check_forbidden_terms(sample: dict, observation: dict, output_text: str, issues: list[str]) -> None:
    answer_text = str(observation.get("answer_text", ""))
    for term in _string_list(sample.get("forbidden_answer_terms", [])):
        if term in answer_text:
            issues.append(f"回答包含禁止身份词: {term}")

    for term in _string_list(sample.get("forbidden_internal_terms", [])):
        if term in output_text:
            issues.append(f"输出包含内部实现词: {term}")


def _combined_output_text(observation: dict) -> str:
    parts = [
        observation.get("answer_text", ""),
        observation.get("process_text", ""),
        observation.get("evidence_text", ""),
        observation.get("raw_text", ""),
    ]
    return "\n".join(str(part) for part in parts if part is not None)


def _string_list(value) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item]


def _string_set(value) -> set[str]:
    return set(_string_list(value))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="输出 Agent 行为评估样本 JSONL")
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLE_PATH, help="评估样本 JSON 文件")
    parser.add_argument("--dry-run", action="store_true", help="只输出样本基线，不调用外部服务")
    args = parser.parse_args(argv)

    samples = load_samples(args.samples)
    write_dry_run(samples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
