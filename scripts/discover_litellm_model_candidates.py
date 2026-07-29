"""从厂商模型快照发现 LiteLLM 候选模型。

本脚本只读取本地 JSON 快照并输出隔离报告，不包含任何 LiteLLM 管理 API
写入能力，也不会修改 Fusion virtual key allowlist。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence


@dataclass(frozen=True)
class ModelCandidate:
    provider_key: str
    provider_display: str
    model_id: str
    litellm_model: str
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class UnknownCandidate:
    source: str
    model_id: str
    reason: str


@dataclass(frozen=True)
class CandidateReport:
    provider_key: str
    provider_display: str
    new: list[ModelCandidate] = field(default_factory=list)
    existing: list[ModelCandidate] = field(default_factory=list)
    removed: list[ModelCandidate] = field(default_factory=list)
    unknown: list[UnknownCandidate] = field(default_factory=list)


class ProviderAdapter(Protocol):
    """把厂商快照和 LiteLLM 条目映射到统一模型 id。"""

    provider_key: str
    provider_display: str

    def source_model_id(self, entry: Mapping[str, Any]) -> str: ...

    def adapt_upstream_model(self, entry: Mapping[str, Any]) -> tuple[ModelCandidate | None, str | None]: ...

    def owns_litellm_entry(self, entry: Mapping[str, Any]) -> bool: ...

    def model_id_from_litellm(self, entry: Mapping[str, Any]) -> str | None: ...

    def build_candidate(self, model_id: str, aliases: Sequence[str] = ()) -> ModelCandidate: ...


class OpenAICompatibleProviderAdapter:
    """适配标准 OpenAI-compatible `/v1/models` 响应。"""

    def __init__(
        self,
        *,
        provider_key: str,
        provider_display: str,
        litellm_prefix: str,
        api_model_prefix: str = "",
        upstream_id_field: str = "id",
        upstream_strip_prefix: str = "",
        required_generation_method: str = "",
    ) -> None:
        self.provider_key = provider_key.strip().lower()
        self.provider_display = provider_display.strip()
        self.litellm_prefix = litellm_prefix.strip().strip("/")
        self.api_model_prefix = api_model_prefix.strip()
        self.upstream_id_field = upstream_id_field.strip()
        self.upstream_strip_prefix = upstream_strip_prefix.strip()
        self.required_generation_method = required_generation_method.strip()
        if not self.provider_key or not self.provider_display or not self.litellm_prefix or not self.upstream_id_field:
            raise ValueError("provider_key、provider_display、litellm_prefix 和 upstream_id_field 不能为空")

    def source_model_id(self, entry: Mapping[str, Any]) -> str:
        return _nonempty_string(entry.get(self.upstream_id_field))

    def adapt_upstream_model(self, entry: Mapping[str, Any]) -> tuple[ModelCandidate | None, str | None]:
        model_id = self.source_model_id(entry)
        if not model_id:
            return None, f"模型条目缺少非空 {self.upstream_id_field}"
        if self.upstream_strip_prefix:
            if not model_id.startswith(self.upstream_strip_prefix):
                return None, f"模型 id 不符合待移除前缀 {self.upstream_strip_prefix}"
            model_id = _nonempty_string(model_id[len(self.upstream_strip_prefix) :])
            if not model_id:
                return None, "模型 id 去除前缀后为空"
        if self.api_model_prefix and not model_id.startswith(self.api_model_prefix):
            return None, f"模型 id 不符合前缀 {self.api_model_prefix}"
        if self.required_generation_method:
            generation_methods = entry.get("supportedGenerationMethods")
            if not isinstance(generation_methods, list) or self.required_generation_method not in generation_methods:
                return None, f"模型不支持 {self.required_generation_method}"
        return self.build_candidate(model_id), None

    def owns_litellm_entry(self, entry: Mapping[str, Any]) -> bool:
        metadata = _entry_metadata(entry)
        metadata_provider = _nonempty_string(metadata.get("provider_key")).lower()
        if metadata_provider:
            return metadata_provider == self.provider_key
        underlying = _entry_underlying_model(entry)
        model_id = self._strip_litellm_prefix(underlying)
        if not model_id:
            return False
        if self.api_model_prefix:
            return model_id.startswith(self.api_model_prefix)
        # `openai/*` 常被多个兼容厂商共用；没有 metadata 或模型前缀时不能安全认领。
        return self.litellm_prefix.lower() == self.provider_key

    def model_id_from_litellm(self, entry: Mapping[str, Any]) -> str | None:
        return self._strip_litellm_prefix(_entry_underlying_model(entry))

    def build_candidate(self, model_id: str, aliases: Sequence[str] = ()) -> ModelCandidate:
        return ModelCandidate(
            provider_key=self.provider_key,
            provider_display=self.provider_display,
            model_id=model_id,
            litellm_model=f"{self.litellm_prefix}/{model_id}",
            aliases=tuple(sorted(set(aliases))),
        )

    def _strip_litellm_prefix(self, underlying: str) -> str | None:
        expected_prefix = f"{self.litellm_prefix}/"
        if not underlying.startswith(expected_prefix):
            return None
        return _nonempty_string(underlying[len(expected_prefix) :]) or None


class MoonshotProviderAdapter(OpenAICompatibleProviderAdapter):
    """Moonshot 专用适配器，只接纳 Kimi 对话模型。"""

    def __init__(self) -> None:
        super().__init__(
            provider_key="moonshot",
            provider_display="Moonshot",
            litellm_prefix="moonshot",
            api_model_prefix="kimi-",
        )

    def adapt_upstream_model(self, entry: Mapping[str, Any]) -> tuple[ModelCandidate | None, str | None]:
        model_id = _nonempty_string(entry.get("id"))
        if not model_id:
            return None, "模型条目缺少非空 id"
        if not model_id.startswith(self.api_model_prefix):
            return None, "Moonshot 适配器不接纳非 Kimi 模型"
        return self.build_candidate(model_id), None


def _nonempty_string(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _entry_metadata(entry: Mapping[str, Any]) -> Mapping[str, Any]:
    model_info = entry.get("model_info") or {}
    if not isinstance(model_info, Mapping):
        return {}
    metadata = model_info.get("metadata") or {}
    return metadata if isinstance(metadata, Mapping) else {}


def _entry_underlying_model(entry: Mapping[str, Any]) -> str:
    params = entry.get("litellm_params") or {}
    if not isinstance(params, Mapping):
        return ""
    return _nonempty_string(params.get("model"))


def _snapshot_entries(snapshot: Any, *, source: str) -> tuple[list[Mapping[str, Any]], list[UnknownCandidate]]:
    raw_entries = snapshot.get("data") if isinstance(snapshot, Mapping) else snapshot
    if not isinstance(raw_entries, list):
        raise ValueError(f"{source} 快照必须是列表或包含 data 列表的对象")
    entries: list[Mapping[str, Any]] = []
    unknown: list[UnknownCandidate] = []
    for index, entry in enumerate(raw_entries):
        if isinstance(entry, Mapping):
            entries.append(entry)
        else:
            unknown.append(
                UnknownCandidate(
                    source=source,
                    model_id=f"index:{index}",
                    reason="模型条目不是 JSON 对象",
                )
            )
    return entries, unknown


def _discover_upstream(
    adapter: ProviderAdapter,
    snapshot: Any,
) -> tuple[dict[str, ModelCandidate], list[UnknownCandidate]]:
    entries, unknown = _snapshot_entries(snapshot, source="upstream")
    candidates: dict[str, ModelCandidate] = {}
    for entry in entries:
        candidate, reason = adapter.adapt_upstream_model(entry)
        if candidate is None:
            unknown.append(
                UnknownCandidate(
                    source="upstream",
                    model_id=adapter.source_model_id(entry),
                    reason=reason or "无法映射",
                )
            )
            continue
        if candidate.model_id in candidates:
            unknown.append(
                UnknownCandidate(
                    source="upstream",
                    model_id=candidate.model_id,
                    reason="上游快照包含重复模型 id",
                )
            )
            continue
        candidates[candidate.model_id] = candidate
    return candidates, unknown


def _discover_litellm(
    adapter: ProviderAdapter,
    snapshot: Any,
) -> tuple[dict[str, list[str]], list[UnknownCandidate]]:
    entries, unknown = _snapshot_entries(snapshot, source="litellm")
    aliases_by_model: dict[str, list[str]] = {}
    for entry in entries:
        model_info = entry.get("model_info")
        model_info = model_info if isinstance(model_info, Mapping) else {}
        metadata = model_info.get("metadata")
        metadata = metadata if isinstance(metadata, Mapping) else {}
        if model_info.get("db_model") is not True and str(metadata.get("purpose") or "").startswith("candidate_"):
            continue
        if not adapter.owns_litellm_entry(entry):
            continue
        alias = _nonempty_string(entry.get("model_name"))
        model_id = adapter.model_id_from_litellm(entry)
        if not model_id:
            unknown.append(
                UnknownCandidate(
                    source="litellm",
                    model_id=alias,
                    reason="LiteLLM 条目无法映射到底层厂商模型 id",
                )
            )
            continue
        aliases_by_model.setdefault(model_id, [])
        if alias and alias not in aliases_by_model[model_id]:
            aliases_by_model[model_id].append(alias)
    return aliases_by_model, unknown


def _all_litellm_alias_targets(snapshot: Any) -> dict[str, set[str]]:
    entries, _ = _snapshot_entries(snapshot, source="litellm")
    targets: dict[str, set[str]] = {}
    for entry in entries:
        alias = _nonempty_string(entry.get("model_name"))
        underlying = _entry_underlying_model(entry)
        if alias and underlying:
            targets.setdefault(alias, set()).add(underlying)
    return targets


def discover_candidates(
    *,
    adapter: ProviderAdapter,
    upstream_snapshot: Any,
    litellm_snapshot: Any,
) -> CandidateReport:
    """对比厂商快照与 LiteLLM 目录，返回只读候选分类。"""
    upstream, upstream_unknown = _discover_upstream(adapter, upstream_snapshot)
    litellm, litellm_unknown = _discover_litellm(adapter, litellm_snapshot)
    if not upstream:
        return CandidateReport(
            provider_key=adapter.provider_key,
            provider_display=adapter.provider_display,
            unknown=[
                *upstream_unknown,
                UnknownCandidate(
                    source="upstream",
                    model_id="",
                    reason="上游可识别模型列表为空，跳过 removed 判定",
                ),
                *litellm_unknown,
            ],
        )
    upstream_ids = set(upstream)
    litellm_ids = set(litellm)
    alias_targets = _all_litellm_alias_targets(litellm_snapshot)
    new_ids = upstream_ids - litellm_ids
    existing_ids = upstream_ids & litellm_ids
    alias_conflicts: list[UnknownCandidate] = []
    for model_id in sorted(new_ids):
        targets = alias_targets.get(model_id, set())
        if not targets:
            continue
        alias_conflicts.append(
            UnknownCandidate(
                source="litellm",
                model_id=model_id,
                reason="候选业务别名已被未归属当前 provider 的 LiteLLM 条目占用",
            )
        )
        new_ids.remove(model_id)

    new = [upstream[model_id] for model_id in sorted(new_ids)]
    existing = [adapter.build_candidate(model_id, litellm.get(model_id, [])) for model_id in sorted(existing_ids)]
    removed = [adapter.build_candidate(model_id, litellm[model_id]) for model_id in sorted(litellm_ids - upstream_ids)]
    return CandidateReport(
        provider_key=adapter.provider_key,
        provider_display=adapter.provider_display,
        new=new,
        existing=existing,
        removed=removed,
        unknown=[*upstream_unknown, *litellm_unknown, *alias_conflicts],
    )


def _serialize_candidate(candidate: ModelCandidate, *, category: str) -> dict[str, Any]:
    isolation_status = {
        "new": "candidate",
        "existing": "observed",
        "removed": "retirement_review",
    }[category]
    return {
        **asdict(candidate),
        "aliases": list(candidate.aliases),
        "isolation_status": isolation_status,
        "eligible_for_registration": False,
    }


def serialize_report(report: CandidateReport) -> dict[str, Any]:
    """序列化稳定、显式只读的候选隔离报告。"""
    return {
        "mode": "read_only",
        "writes_performed": False,
        "write_actions": [],
        "provider": {
            "key": report.provider_key,
            "display": report.provider_display,
        },
        "summary": {
            "new": len(report.new),
            "existing": len(report.existing),
            "removed": len(report.removed),
            "unknown": len(report.unknown),
        },
        "new": [_serialize_candidate(item, category="new") for item in report.new],
        "existing": [_serialize_candidate(item, category="existing") for item in report.existing],
        "removed": [_serialize_candidate(item, category="removed") for item in report.removed],
        "unknown": [asdict(item) for item in report.unknown],
    }


def _build_adapter(args: argparse.Namespace) -> ProviderAdapter:
    if args.provider == "moonshot":
        return MoonshotProviderAdapter()
    missing = [
        name
        for name, value in {
            "--provider-key": args.provider_key,
            "--provider-display": args.provider_display,
            "--litellm-prefix": args.litellm_prefix,
        }.items()
        if not value
    ]
    if missing:
        raise ValueError(f"openai-compatible 适配器缺少参数: {', '.join(missing)}")
    return OpenAICompatibleProviderAdapter(
        provider_key=args.provider_key,
        provider_display=args.provider_display,
        litellm_prefix=args.litellm_prefix,
        api_model_prefix=args.api_model_prefix,
    )


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="只读发现 LiteLLM 模型候选")
    parser.add_argument("--provider", choices=("moonshot", "openai-compatible"), required=True)
    parser.add_argument("--upstream-snapshot", type=Path, required=True)
    parser.add_argument("--litellm-snapshot", type=Path, required=True)
    parser.add_argument("--provider-key", default="")
    parser.add_argument("--provider-display", default="")
    parser.add_argument("--litellm-prefix", default="")
    parser.add_argument("--api-model-prefix", default="")
    return parser.parse_args(argv)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取 JSON 快照 {path}: {exc}") from exc


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    adapter = _build_adapter(args)
    report = discover_candidates(
        adapter=adapter,
        upstream_snapshot=_read_json(args.upstream_snapshot),
        litellm_snapshot=_read_json(args.litellm_snapshot),
    )
    print(json.dumps(serialize_report(report), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
