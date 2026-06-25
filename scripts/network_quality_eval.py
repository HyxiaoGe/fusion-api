"""
联网搜索质量评估集 V1。

默认 dry-run 只校验固定样本并输出 JSONL，不调用真实 search-service。
后续接入真实搜索结果时，可复用 load_samples() 和 score_results() 做对比。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TextIO
from urllib.parse import urlsplit

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SAMPLE_PATH = PROJECT_ROOT / "test" / "fixtures" / "network_quality_eval_samples.json"

REQUIRED_SAMPLE_FIELDS = {"id", "category", "question", "query", "intent", "expected_domains"}


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

        expected_domains = item.get("expected_domains")
        if not isinstance(expected_domains, list) or not expected_domains:
            raise ValueError(f"样本 expected_domains 必须是非空数组: id={sample_id}")

        samples.append(item)

    return samples


def score_results(sample: dict, results: list[dict]) -> dict:
    expected_domains = _normalize_domain_list(sample.get("expected_domains", []))
    official_domains = _normalize_domain_list(sample.get("official_domains", []))
    min_results = _positive_int(sample.get("min_results"), default=5)
    result_domains = [_extract_domain(str(result.get("url", ""))) for result in results if isinstance(result, dict)]
    result_domains = [domain for domain in result_domains if domain]

    unique_domains = sorted(set(result_domains))
    duplicate_domain_count = max(0, len(result_domains) - len(unique_domains))
    expected_hits = _matched_expectations(result_domains, expected_domains)
    official_hits = _matched_expectations(result_domains, official_domains)

    count_score = min(len(results), min_results) / min_results * 20
    expected_score = (len(expected_hits) / len(expected_domains) * 30) if expected_domains else 0
    official_score = (len(official_hits) / len(official_domains) * 20) if official_domains else 0
    diversity_score = (len(unique_domains) / len(result_domains) * 30) if result_domains else 0
    duplicate_penalty = min(20, duplicate_domain_count * 5)
    total_score = max(0, min(100, round(count_score + expected_score + official_score + diversity_score - duplicate_penalty)))

    return {
        "score": total_score,
        "result_count": len(results),
        "unique_domain_count": len(unique_domains),
        "duplicate_domain_count": duplicate_domain_count,
        "expected_domain_hits": expected_hits,
        "official_domain_hits": official_hits,
    }


def write_dry_run(samples: list[dict], output: TextIO = sys.stdout) -> None:
    for sample in samples:
        score = score_results(sample, [])
        output.write(
            json.dumps(
                {
                    "sample_id": sample["id"],
                    "category": sample["category"],
                    "query": sample["query"],
                    "intent": sample["intent"],
                    "expected_domains": sample["expected_domains"],
                    "official_domains": sample.get("official_domains", []),
                    "min_results": sample.get("min_results", 5),
                    **score,
                },
                ensure_ascii=False,
            )
        )
        output.write("\n")


def _normalize_domain_list(values: list) -> list[str]:
    normalized = []
    for value in values:
        if not isinstance(value, str):
            continue
        domain = _normalize_domain(value)
        if domain:
            normalized.append(domain)
    return sorted(set(normalized))


def _matched_expectations(result_domains: list[str], expected_domains: list[str]) -> list[str]:
    matched: set[str] = set()
    specific_first = sorted(expected_domains, key=len, reverse=True)
    for domain in result_domains:
        for expected in specific_first:
            if _domain_matches(domain, expected):
                matched.add(expected)
                break
    return [expected for expected in expected_domains if expected in matched]


def _extract_domain(url: str) -> str:
    try:
        parsed = urlsplit(url)
    except ValueError:
        return ""
    return _normalize_domain(parsed.hostname or "")


def _normalize_domain(domain: str) -> str:
    normalized = domain.strip().rstrip(".").lower()
    while normalized.startswith("www."):
        normalized = normalized[4:]
    return normalized


def _domain_matches(domain: str, expected: str) -> bool:
    return domain == expected or domain.endswith(f".{expected}")


def _positive_int(value, *, default: int) -> int:
    return value if isinstance(value, int) and value > 0 else default


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="输出联网搜索质量评估样本 JSONL")
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLE_PATH, help="评估样本 JSON 文件")
    parser.add_argument("--dry-run", action="store_true", help="只输出样本基线，不调用搜索服务")
    args = parser.parse_args(argv)

    samples = load_samples(args.samples)
    write_dry_run(samples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
