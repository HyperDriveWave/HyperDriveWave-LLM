from __future__ import annotations

import base64
import math
from dataclasses import dataclass
from typing import Any, Awaitable, Callable


IMAGE_TOKEN_ESTIMATE = 256


def estimate_tokens(text: str) -> int:
    """Conservative mixed Chinese/Latin estimate; exact tokenizer is provider-owned."""
    chinese = sum("\u4e00" <= char <= "\u9fff" for char in text)
    other = max(0, len(text) - chinese)
    return max(1, chinese + math.ceil(other / 4))


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(item.get("text"))
            for item in content
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        )
    if isinstance(content, dict) and isinstance(content.get("text"), str):
        return str(content["text"])
    return ""


def _image_payload_stats(content: Any) -> tuple[int, int]:
    if not isinstance(content, list):
        return 0, 0
    count = 0
    byte_count = 0
    for item in content:
        if not isinstance(item, dict):
            continue
        image_url = item.get("image_url")
        url = image_url.get("url") if isinstance(image_url, dict) else ""
        if item.get("type") not in {"image", "image_url"} or not isinstance(url, str):
            continue
        count += 1
        if url.startswith("data:") and "," in url:
            encoded = url.split(",", 1)[1].strip()
            padding = len(encoded) - len(encoded.rstrip("="))
            byte_count += max(0, (len(encoded) * 3 // 4) - padding)
    return count, byte_count


def media_stats(messages: list[dict[str, Any]]) -> dict[str, int]:
    image_count = 0
    image_bytes = 0
    for message in messages:
        count, byte_count = _image_payload_stats(message.get("content"))
        image_count += count
        image_bytes += byte_count
    return {
        "image_count": image_count,
        "image_bytes": image_bytes,
        "image_token_estimate": image_count * IMAGE_TOKEN_ESTIMATE,
    }


def estimate_messages(messages: list[dict[str, Any]]) -> int:
    total = 0
    for message in messages:
        text = _content_text(message.get("content"))
        image_count, _ = _image_payload_stats(message.get("content"))
        total += estimate_tokens(text) + 4 + image_count * IMAGE_TOKEN_ESTIMATE
    return total


@dataclass(frozen=True)
class ContextPreparation:
    messages: list[dict[str, Any]]
    compressed: bool
    tokens_before: int
    tokens_after: int
    kept_from: int
    summary: str
    image_count: int = 0
    image_bytes: int = 0
    image_token_estimate: int = 0


class ContextManager:
    def __init__(
        self,
        *,
        window_tokens: int,
        threshold: float = 0.75,
        target: float = 0.55,
    ) -> None:
        self.window_tokens = max(1, window_tokens)
        self.threshold_tokens = max(1, int(self.window_tokens * threshold))
        self.target_tokens = max(1, int(self.window_tokens * target))

    async def prepare(
        self,
        messages: list[dict[str, Any]],
        *,
        existing_summary: str = "",
        existing_kept_from: int = 0,
        fixed_messages: list[dict[str, Any]] | None = None,
        extra_messages: list[dict[str, Any]] | None = None,
        summarize: Callable[[list[dict[str, Any]]], Awaitable[str]],
    ) -> ContextPreparation:
        fixed_messages = fixed_messages or []
        extra_messages = extra_messages or []
        kept_from = max(0, min(existing_kept_from, len(messages)))
        prefix = (
            [{"role": "system", "content": f"此前会话摘要：\n{existing_summary}"}]
            if existing_summary
            else []
        )
        active = prefix + messages[kept_from:] + fixed_messages + extra_messages
        tokens_before = estimate_messages(active)
        if tokens_before < self.threshold_tokens or not messages[kept_from:]:
            compact = prefix + messages[kept_from:]
            return ContextPreparation(
                messages=compact,
                compressed=False,
                tokens_before=tokens_before,
                tokens_after=estimate_messages(
                    compact + fixed_messages + extra_messages
                ),
                kept_from=kept_from,
                summary=existing_summary,
                **media_stats(compact + fixed_messages + extra_messages),
            )

        recent: list[dict[str, str]] = []
        fixed_tokens = estimate_messages(fixed_messages + extra_messages)
        recent_budget = max(0, self.target_tokens - fixed_tokens)
        recent_tokens = 0
        cut = len(messages)
        for index in range(len(messages) - 1, kept_from - 1, -1):
            candidate = messages[index]
            candidate_tokens = estimate_messages([candidate])
            if recent and recent_tokens + candidate_tokens > recent_budget:
                break
            recent.insert(0, candidate)
            recent_tokens += candidate_tokens
            cut = index

        summarized_messages = prefix + messages[kept_from:cut]
        summary = (await summarize(summarized_messages)).strip()
        if not summary:
            raise RuntimeError("context compression returned an empty summary")

        compact_prefix = [{"role": "system", "content": f"此前会话摘要：\n{summary}"}]
        compact = compact_prefix + recent
        return ContextPreparation(
            messages=compact,
            compressed=True,
            tokens_before=tokens_before,
            tokens_after=estimate_messages(
                compact + fixed_messages + extra_messages
            ),
            kept_from=cut,
            summary=summary,
            **media_stats(compact + fixed_messages + extra_messages),
        )


async def _self_check() -> None:
    assert estimate_tokens("你好") == 2
    assert estimate_tokens("abcd") == 1
    manager = ContextManager(window_tokens=100, threshold=0.75, target=0.5)
    result = await manager.prepare(
        [{"role": "user", "content": "旧问题 " * 20}],
        extra_messages=[{"role": "user", "content": "新问题"}],
        fixed_messages=[{"role": "system", "content": "工业问答提示"}],
        summarize=lambda _: _summary(),
    )
    assert result.compressed
    assert result.summary == "保留的摘要"
    assert result.tokens_after > 0
    image = base64.b64encode(b"x" * 10000).decode("ascii")
    multimodal = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "请分析图片"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}},
            ],
        }
    ]
    assert estimate_messages(multimodal) < 1000
    assert media_stats(multimodal)["image_bytes"] == 10000


async def _summary() -> str:
    return "保留的摘要"


if __name__ == "__main__":
    import asyncio

    asyncio.run(_self_check())
    print("context manager self-check passed")
