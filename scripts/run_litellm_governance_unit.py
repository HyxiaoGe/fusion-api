"""安全加载 LiteLLM 治理任务环境并执行目标命令。

systemd 单元不得直接使用 EnvironmentFile：它会在 ExecStartPre 前加载全部变量。
本 wrapper 先检查 secret 文件，再按白名单解析并以最小环境执行目标命令。
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import stat
import sys
from pathlib import Path
from typing import Mapping, Sequence

from scripts.check_litellm_governance_runtime import runtime_status

ENV_NAME_PATTERN = re.compile(r"^[A-Z_][A-Z0-9_]*$")
GOVERNANCE_ENV_NAMES = {
    "FUSION_MODEL_MANAGEMENT_BASE_URL",
    "LITELLM_MODEL_ADMISSION_WORKER_TOKEN",
    "LITELLM_BASE_URL",
    "LITELLM_PROXY_URL",
    "LITELLM_CANDIDATE_KEY",
    "LITELLM_VIRTUAL_KEY",
}
SAFE_INHERITED_ENV_NAMES = {
    "HOME",
    "PATH",
    "LANG",
    "LC_ALL",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
}


def _read_registry_api_key_envs(payload_text: str | None) -> set[str]:
    if payload_text is None:
        return set()
    payload = json.loads(payload_text)
    providers = payload.get("providers") if isinstance(payload, Mapping) else None
    if not isinstance(providers, list):
        raise ValueError("registry.providers 必须是列表")
    names: set[str] = set()
    for provider in providers:
        name = provider.get("api_key_env") if isinstance(provider, Mapping) else None
        if (
            not isinstance(name, str)
            or not ENV_NAME_PATTERN.fullmatch(name)
            or not name.endswith("_API_KEY")
            or name == "LITELLM_CANDIDATE_KEY"
        ):
            raise ValueError("provider api_key_env 无效")
        names.add(name)
    return names


def _read_secure_text(path: Path) -> str:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            raise ValueError("secret file 不是普通文件")
        if file_stat.st_uid != os.getuid():
            raise ValueError("secret file owner 不匹配")
        if stat.S_IMODE(file_stat.st_mode) & 0o077:
            raise ValueError("secret file 权限过宽")
        with os.fdopen(descriptor, encoding="utf-8") as handle:
            descriptor = -1
            return handle.read()
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        _read_secure_text(path).splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").lstrip()
        if "=" not in line:
            raise ValueError(f"环境文件第 {line_number} 行缺少等号")
        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip()
        if not ENV_NAME_PATTERN.fullmatch(name):
            raise ValueError(f"环境文件第 {line_number} 行变量名无效")
        if name in values:
            raise ValueError(f"环境文件变量重复: {name}")
        if value.startswith(("'", '"')):
            quote = value[0]
            if len(value) < 2 or not value.endswith(quote):
                raise ValueError(f"环境文件第 {line_number} 行引号未闭合")
            value = value[1:-1]
        values[name] = value
    return values


def build_exec_environment(
    *,
    proxy_env: Path,
    governance_env: Path,
    registry: Path | None,
    required_env_names: Sequence[str],
    inherited: Mapping[str, str],
) -> tuple[dict[str, str], list[str], str | None]:
    """在 secret 文件已通过 runtime_status 后构建最小执行环境。"""
    registry_text = _read_secure_text(registry) if registry is not None else None
    provider_names = _read_registry_api_key_envs(registry_text)
    proxy_values = _parse_env_file(proxy_env)
    governance_values = _parse_env_file(governance_env)
    issues: list[str] = []

    forbidden_governance_names = {
        name
        for name in governance_values
        if name == "LITELLM_MASTER_KEY" or (name.endswith("_API_KEY") and name != "LITELLM_CANDIDATE_KEY")
    }
    if forbidden_governance_names:
        issues.append("governance_env_duplicates_upstream_credentials")

    allowed_proxy_names = {"LITELLM_MASTER_KEY", *provider_names}
    selected = {name: value for name, value in proxy_values.items() if name in allowed_proxy_names}
    selected.update({name: value for name, value in governance_values.items() if name in GOVERNANCE_ENV_NAMES})
    for name in {*required_env_names, *provider_names}:
        if not selected.get(name):
            issues.append("required_environment_missing")

    execute_env = {name: value for name, value in inherited.items() if name in SAFE_INHERITED_ENV_NAMES}
    execute_env["PYTHONNOUSERSITE"] = "1"
    execute_env.update(selected)
    return execute_env, list(dict.fromkeys(issues)), registry_text


def _materialize_registry_snapshot(payload_text: str) -> tuple[int, str]:
    required_features = (
        hasattr(os, "memfd_create"),
        hasattr(os, "MFD_ALLOW_SEALING"),
        hasattr(fcntl, "F_ADD_SEALS"),
        hasattr(fcntl, "F_SEAL_WRITE"),
    )
    if not all(required_features):
        raise RuntimeError("当前平台不支持 sealed memfd")
    descriptor = os.memfd_create(
        "litellm-provider-registry",
        flags=os.MFD_ALLOW_SEALING,
    )
    try:
        payload = payload_text.encode("utf-8")
        written = 0
        while written < len(payload):
            written += os.write(descriptor, payload[written:])
        seals = fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL
        fcntl.fcntl(descriptor, fcntl.F_ADD_SEALS, seals)
        os.lseek(descriptor, 0, os.SEEK_SET)
        os.set_inheritable(descriptor, True)
        return descriptor, f"/proc/self/fd/{descriptor}"
    except Exception:
        os.close(descriptor)
        raise


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy-env", type=Path, required=True)
    parser.add_argument("--governance-env", type=Path, required=True)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--require-env", action="append", default=[])
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("必须在 -- 后提供目标命令")
    if not Path(args.command[0]).is_absolute():
        parser.error("目标命令必须使用绝对路径")
    return args


def _print_failure(issues: Sequence[str]) -> None:
    print(
        json.dumps(
            {"healthy": False, "issues": list(dict.fromkeys(issues))},
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    status = runtime_status()
    if not status["healthy"]:
        _print_failure(status["issues"])
        return 1
    try:
        execute_env, issues, registry_text = build_exec_environment(
            proxy_env=args.proxy_env,
            governance_env=args.governance_env,
            registry=args.registry,
            required_env_names=args.require_env,
            inherited=os.environ,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        _print_failure([f"environment_load_failed:{type(exc).__name__}"])
        return 1
    if issues:
        _print_failure(issues)
        return 1
    registry_descriptor: int | None = None
    command = list(args.command)
    try:
        if registry_text is not None and args.registry is not None:
            registry_descriptor, registry_snapshot_path = _materialize_registry_snapshot(registry_text)
            command = [registry_snapshot_path if value == str(args.registry) else value for value in command]
        os.execve(command[0], command, execute_env)
        raise AssertionError("os.execve 不应返回")
    except (OSError, RuntimeError) as exc:
        _print_failure([f"exec_failed:{type(exc).__name__}"])
        return 1
    finally:
        if registry_descriptor is not None:
            os.close(registry_descriptor)


if __name__ == "__main__":
    raise SystemExit(main())
