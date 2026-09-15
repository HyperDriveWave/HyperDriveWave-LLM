from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any

import httpx

# 抑制模型复读检索到的原文。复读会让输出冲到 max_tokens 上限，并触发
# llama.cpp 的 chat 解析失败。这是标定值：调大更不易复读但更易跑题。
_PRESENCE_PENALTY = float(os.getenv("HDW_LLM_PRESENCE_PENALTY", "0.3"))


@dataclass(frozen=True)
class LLMResult:
    content: str
    reasoning_content: str = ""
    # 上游的 finish_reason（"stop" / "length" / ...）。默认空串保持向后兼容——
    # 这个字段是后加的，任何按位置构造 LLMResult 的调用方都不该因此坏掉。
    #
    # 为什么需要它：**被 max_tokens 截断的输出看起来和正常输出一模一样**，
    # 只是短了。图片转写场景下这意味着「整张卷子后半部分的题全丢了」却毫无征兆，
    # 下游拿着半份题面去检索、去作答，谁都不知道少了东西。
    finish_reason: str = ""


class OpenAICompatibleClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str = "",
        timeout: float | None = 120,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key.strip()
        self.timeout = timeout

    @property
    def completion_url(self) -> str:
        return f"{self.base_url}/chat/completions"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int,
        temperature: float = 0.2,
        reasoning_effort: str | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
        thinking_budget_tokens: int | None = None,
        thinking_enabled: bool | None = None,
    ) -> LLMResult:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
            "presence_penalty": _PRESENCE_PENALTY,
        }
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort
        if chat_template_kwargs:
            payload["chat_template_kwargs"] = chat_template_kwargs
        if thinking_budget_tokens is not None:
            payload["thinking_budget_tokens"] = max(0, int(thinking_budget_tokens))
        if thinking_enabled is not None:
            payload["thinking"] = {
                "type": "enabled" if thinking_enabled else "disabled",
            }

        # 上游（llama.cpp）在请求来得比它消化得快时会回一个**空的 400**
        # （无 body、`connection: close`、服务端日志里连任务都没创建）。
        #
        # 实测把触发条件量出来了：
        #     间隔 3 秒发 20 次 → 0/20 失败
        #     不间隔连发 20 次  → 9/20 失败（45%）
        # 所以它是**速率**问题，不是"偶发"——原来那句「偶发瞬时 4xx/5xx」
        # 的判断是错的，因而重试只隔 0.5s 也不够（0.5s 仍落在过快的那一侧）。
        #
        # 改成指数退避 + 三次尝试：第一次隔 1s、第二次隔 2s。
        # 1s 起步是因为要跨过"比服务器消化得快"的那条线，而不是消除网络抖动。
        _LLM_RETRY_BACKOFF = (1.0, 2.0)
        attempts = len(_LLM_RETRY_BACKOFF) + 1
        for attempt in range(attempts):
            if attempt:
                await asyncio.sleep(_LLM_RETRY_BACKOFF[attempt - 1])
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        self.completion_url,
                        headers=self._headers(),
                        json=payload,
                    )
            except httpx.HTTPError:
                if attempt == attempts - 1:
                    raise
                continue
            if not response.is_error:
                break
            if attempt == attempts - 1:
                raise RuntimeError(f"LLM HTTP {response.status_code}: {response.text[:500]}")

        try:
            choice = response.json()["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise RuntimeError("LLM response does not contain a message") from exc
        reasoning_content = str(message.get("reasoning_content") or "").strip()
        # Some Qwen-compatible servers return reasoning only in this channel.
        content = str(message.get("content") or "").strip() or reasoning_content
        if not content:
            raise RuntimeError("LLM returned empty content")
        return LLMResult(
            content=content,
            reasoning_content=reasoning_content,
            # 上游不给这个字段时留空串（不是 None），调用方按「未知」处理。
            finish_reason=str(choice.get("finish_reason") or "").strip(),
        )

    async def health(self) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(
                f"{self.base_url}/models",
                headers=self._headers(),
            )
        response.raise_for_status()
        return response.json()
