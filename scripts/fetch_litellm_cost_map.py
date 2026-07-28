"""只读拉取并校验 LiteLLM 官方成本表，原子保存数据与状态摘要。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import httpx

DEFAULT_COST_MAP_URL = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"


def _is_mapping(value: Any) -> bool:
    return isinstance(value, Mapping)


def validate_cost_map(payload: Any) -> Mapping[str, Any]:
    if not _is_mapping(payload) or not payload:
        raise ValueError("LiteLLM 成本表必须是非空 JSON 对象")
    priced = sum(
        1
        for entry in payload.values()
        if _is_mapping(entry)
        and isinstance(entry.get("input_cost_per_token"), (int, float))
        and not isinstance(entry.get("input_cost_per_token"), bool)
    )
    if priced == 0:
        raise ValueError("LiteLLM 成本表没有可识别的 input_cost_per_token")
    return payload


def fetch_cost_map(*, url: str, client: Any = httpx, timeout_seconds: float = 30.0) -> tuple[Mapping[str, Any], Any]:
    response = client.get(url, timeout=timeout_seconds, follow_redirects=True)
    response.raise_for_status()
    return validate_cost_map(response.json()), response


def cost_map_sha256(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def build_status(*, payload: Mapping[str, Any], response: Any, source_url: str) -> dict[str, Any]:
    headers = getattr(response, "headers", {})
    return {
        "status": "success",
        "source_url": source_url,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "model_count": len(payload),
        "sha256": cost_map_sha256(payload),
        "etag": headers.get("etag"),
        "last_modified": headers.get("last-modified"),
    }


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temp_path, path)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_COST_MAP_URL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--status-output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    try:
        payload, response = fetch_cost_map(url=args.url, timeout_seconds=args.timeout_seconds)
        status = build_status(payload=payload, response=response, source_url=args.url)
        write_json_atomic(args.output, payload)
        write_json_atomic(args.status_output, status)
    except (httpx.HTTPError, json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
        print(
            json.dumps(
                {"status": "error", "error": {"code": "cost_map_fetch_failed", "type": type(exc).__name__}},
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(status, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
