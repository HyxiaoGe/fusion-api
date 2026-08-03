"""真实模型调用前的可见输出语言策略。"""

from __future__ import annotations

from app.ai.prompts.agent_loop import VISIBLE_RESPONSE_LANGUAGE_PROMPT


def finalize_model_call_language_policy(messages: list[dict]) -> list[dict]:
    """在最后一条有效 system 指令末尾保留唯一语言契约，不修改原消息。"""

    finalized: list[dict] = []
    for message in messages:
        copied = dict(message)
        if copied.get("role") == "system" and isinstance(copied.get("content"), str):
            content = copied["content"].replace(VISIBLE_RESPONSE_LANGUAGE_PROMPT, "").rstrip()
            if not content:
                continue
            copied["content"] = content
        finalized.append(copied)

    last_system_index = next(
        (index for index in range(len(finalized) - 1, -1, -1) if finalized[index].get("role") == "system"),
        None,
    )
    if last_system_index is None:
        return [
            {"role": "system", "content": VISIBLE_RESPONSE_LANGUAGE_PROMPT},
            *finalized,
        ]

    last_system = finalized[last_system_index]
    content = last_system.get("content")
    if isinstance(content, str):
        last_system["content"] = f"{content.rstrip()}\n\n{VISIBLE_RESPONSE_LANGUAGE_PROMPT}"
        return finalized

    finalized.insert(
        last_system_index + 1,
        {"role": "system", "content": VISIBLE_RESPONSE_LANGUAGE_PROMPT},
    )
    return finalized
