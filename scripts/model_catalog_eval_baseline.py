"""对 Fusion 可选模型执行统一 smoke，生成 JSONL 基线。

脚本默认 dry-run，只列出将被测的模型；显式 `--apply` 才会请求 `/api/chat/send`。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import httpx

DEFAULT_FUSION_BASE_URL = "https://fusion.seanfield.org"
DEFAULT_QUESTION = "请用一句话介绍你能做什么。"


@dataclass(frozen=True)
class EvalResult:
    model_id: str
    provider: str
    model_name: str
    question: str
    success: bool
    elapsed_ms: int
    answer_preview: str
    error: dict[str, str] | None


def select_models(
    models: Sequence[Mapping[str, Any]],
    *,
    include_unhealthy: bool = False,
    model_ids: Sequence[str] | None = None,
) -> list[Mapping[str, Any]]:
    allowed_ids = set(model_ids or [])
    selected: list[Mapping[str, Any]] = []
    for model in models:
        model_id = str(model.get("modelId") or "")
        if allowed_ids and model_id not in allowed_ids:
            continue
        health = model.get("health") or {}
        if not include_unhealthy and health.get("status") == "unhealthy":
            continue
        selected.append(model)
    return selected


def _extract_answer_preview(response_payload: Mapping[str, Any], limit: int = 240) -> str:
    data = response_payload.get("data") or {}
    message = data.get("message") or {}
    content = message.get("content") or data.get("content") or ""
    if isinstance(content, list):
        content = " ".join(
            str(item.get("text") or item.get("content") or "") for item in content if isinstance(item, dict)
        )
    text = str(content).strip().replace("\n", " ")
    return text[:limit]


def build_success_result(
    *,
    model: Mapping[str, Any],
    question: str,
    elapsed_ms: int,
    response_payload: Mapping[str, Any],
) -> EvalResult:
    return EvalResult(
        model_id=str(model.get("modelId") or ""),
        provider=str(model.get("provider") or ""),
        model_name=str(model.get("name") or ""),
        question=question,
        success=True,
        elapsed_ms=elapsed_ms,
        answer_preview=_extract_answer_preview(response_payload),
        error=None,
    )


def build_failure_result(
    *,
    model: Mapping[str, Any],
    question: str,
    elapsed_ms: int,
    error: Exception,
) -> EvalResult:
    return EvalResult(
        model_id=str(model.get("modelId") or ""),
        provider=str(model.get("provider") or ""),
        model_name=str(model.get("name") or ""),
        question=question,
        success=False,
        elapsed_ms=elapsed_ms,
        answer_preview="",
        error={"type": type(error).__name__, "message": str(error)},
    )


def to_jsonl(result: EvalResult) -> str:
    return json.dumps(asdict(result), ensure_ascii=False, sort_keys=True) + "\n"


def fetch_models(base_url: str, auth_token: str | None = None) -> list[dict[str, Any]]:
    headers = {"Authorization": f"Bearer {auth_token}"} if auth_token else {}
    response = httpx.get(f"{base_url.rstrip('/')}/api/models/", headers=headers, timeout=20.0)
    response.raise_for_status()
    payload = response.json()
    return list((payload.get("data") or {}).get("models") or [])


def call_chat_send(
    *,
    base_url: str,
    auth_token: str,
    model_id: str,
    question: str,
) -> dict[str, Any]:
    response = httpx.post(
        f"{base_url.rstrip('/')}/api/chat/send",
        headers={"Authorization": f"Bearer {auth_token}", "Content-Type": "application/json"},
        json={"model_id": model_id, "message": question, "stream": False},
        timeout=90.0,
    )
    response.raise_for_status()
    return dict(response.json())


def run_eval(
    *,
    base_url: str,
    auth_token: str,
    models: Iterable[Mapping[str, Any]],
    question: str,
) -> list[EvalResult]:
    results: list[EvalResult] = []
    for model in models:
        started = time.perf_counter()
        try:
            payload = call_chat_send(
                base_url=base_url,
                auth_token=auth_token,
                model_id=str(model.get("modelId") or ""),
                question=question,
            )
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            results.append(
                build_success_result(
                    model=model,
                    question=question,
                    elapsed_ms=elapsed_ms,
                    response_payload=payload,
                )
            )
        except Exception as exc:
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            results.append(
                build_failure_result(
                    model=model,
                    question=question,
                    elapsed_ms=elapsed_ms,
                    error=exc,
                )
            )
    return results


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 Fusion 多模型 smoke 基线")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="只列出将被测模型（默认）")
    mode.add_argument("--apply", action="store_true", help="实际调用 /api/chat/send")
    parser.add_argument("--base-url", default=DEFAULT_FUSION_BASE_URL)
    parser.add_argument("--auth-token", default="")
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--models", default="", help="逗号分隔的 modelId 白名单")
    parser.add_argument("--include-unhealthy", action="store_true")
    parser.add_argument("--output", default="", help="JSONL 输出文件；为空则输出到 stdout")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    models = fetch_models(args.base_url, args.auth_token or None)
    selected = select_models(
        models,
        include_unhealthy=args.include_unhealthy,
        model_ids=_split_csv(args.models),
    )

    if not args.apply:
        print(
            json.dumps(
                [{"modelId": model.get("modelId"), "provider": model.get("provider")} for model in selected],
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if not args.auth_token:
        raise RuntimeError("实际测验需要 --auth-token")

    results = run_eval(
        base_url=args.base_url,
        auth_token=args.auth_token,
        models=selected,
        question=args.question,
    )

    content = "".join(to_jsonl(result) for result in results)
    if args.output:
        Path(args.output).write_text(content, encoding="utf-8")
    else:
        print(content, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
