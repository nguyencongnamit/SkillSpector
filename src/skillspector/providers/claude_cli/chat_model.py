# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A LangChain chat model backed by the local ``claude`` CLI.

Instead of calling ``api.anthropic.com`` with an API key, this model shells
out to ``claude -p --output-format json`` and reads the assistant text back
from the JSON envelope.  Authentication is therefore whatever the host
``claude`` CLI uses — a Claude **subscription** (Max/Pro) via the macOS
Keychain or a ``CLAUDE_CODE_OAUTH_TOKEN`` — and **no API key is required**.

Trade-offs (intentional, surfaced here so they aren't a surprise):

* Each call spawns a ``claude`` subprocess — heavier and slower than a direct
  HTTPS request.
* A personal subscription has rolling usage caps; firing many calls in
  parallel will hit them.  ``_agenerate`` therefore funnels every concurrent
  call through a module-level semaphore sized by
  ``SKILLSPECTOR_CLAUDE_MAX_CONCURRENCY`` (default 3).
* The CLI does not enforce a JSON schema the way the API's structured-output
  mode does.  :meth:`ChatClaudeCli.with_structured_output` compensates by
  injecting the schema into the prompt and validating / repairing the reply.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
from typing import Any

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableConfig
from pydantic import Field

from skillspector.logging_config import get_logger

logger = get_logger(__name__)

_DEFAULT_TIMEOUT = 600.0
_DEFAULT_MAX_CONCURRENCY = 3

# Lazily-created, per-process throttle shared by every ChatClaudeCli instance so
# a 10-way analyzer fan-out doesn't open 10 simultaneous subscription sessions.
_semaphore: asyncio.Semaphore | None = None


def _get_semaphore() -> asyncio.Semaphore:
    """Return the shared subprocess semaphore (created on first use)."""
    global _semaphore
    if _semaphore is None:
        try:
            limit = int(os.environ.get("SKILLSPECTOR_CLAUDE_MAX_CONCURRENCY", ""))
        except ValueError:
            limit = _DEFAULT_MAX_CONCURRENCY
        _semaphore = asyncio.Semaphore(max(1, limit))
    return _semaphore


def _message_text(content: Any) -> str:
    """Flatten a message's ``content`` (str or content-block list) to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return str(content)


def _split_messages(messages: list[BaseMessage]) -> tuple[str, str]:
    """Render LangChain messages into ``(system_text, user_text)``.

    System messages become the CLI ``--append-system-prompt``; everything
    else is concatenated into the prompt fed on stdin.
    """
    system_parts: list[str] = []
    user_parts: list[str] = []
    for msg in messages:
        text = _message_text(msg.content)
        if msg.type == "system":
            system_parts.append(text)
        else:
            user_parts.append(text)
    return "\n\n".join(system_parts).strip(), "\n\n".join(user_parts).strip()


def _parse_cli_output(stdout: str, stderr: str, returncode: int) -> str:
    """Extract the assistant text from ``claude --output-format json`` output."""
    if returncode != 0 and not stdout.strip():
        raise RuntimeError(
            f"claude CLI exited {returncode}: {stderr.strip() or '(no stderr)'}"
        )
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        # No JSON envelope (e.g. a hard error before the result event).
        if stdout.strip():
            return stdout.strip()
        raise RuntimeError(
            f"claude CLI produced no JSON output. stderr: {stderr.strip() or '(none)'}"
        ) from None

    if isinstance(data, dict):
        if data.get("is_error"):
            raise RuntimeError(f"claude CLI reported an error: {data.get('result', data)}")
        result = data.get("result")
        if isinstance(result, str):
            return result
    raise RuntimeError(f"Unexpected claude CLI JSON shape: {data!r}")


class ChatClaudeCli(BaseChatModel):
    """Minimal chat model that delegates generation to the ``claude`` CLI."""

    claude_model: str = "claude-opus-4-6"
    cli_path: str = Field(default_factory=lambda: os.environ.get("SKILLSPECTOR_CLAUDE_CLI", "claude"))
    timeout: float | None = _DEFAULT_TIMEOUT
    # Kept for parity with other providers; the CLI has no hard output-cap flag.
    max_output_tokens: int | None = None

    @property
    def _llm_type(self) -> str:
        return "claude-cli"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"claude_model": self.claude_model, "cli_path": self.cli_path}

    def _build_command(self, system_text: str) -> list[str]:
        cmd = [
            self.cli_path,
            "-p",
            "--output-format",
            "json",
            "--model",
            self.claude_model,
            "--no-session-persistence",
        ]
        if system_text:
            cmd += ["--append-system-prompt", system_text]
        return cmd

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        system_text, user_text = _split_messages(messages)
        cmd = self._build_command(system_text)
        logger.debug("claude CLI call (model=%s, prompt_chars=%d)", self.claude_model, len(user_text))
        try:
            proc = subprocess.run(
                cmd,
                input=user_text,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"claude CLI timed out after {self.timeout}s") from exc
        text = _parse_cli_output(proc.stdout, proc.stderr, proc.returncode)
        message = AIMessage(content=text)
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        system_text, user_text = _split_messages(messages)
        cmd = self._build_command(system_text)
        logger.debug("claude CLI acall (model=%s, prompt_chars=%d)", self.claude_model, len(user_text))
        async with _get_semaphore():
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout_b, stderr_b = await asyncio.wait_for(
                    proc.communicate(user_text.encode()),
                    timeout=self.timeout,
                )
            except TimeoutError as exc:
                proc.kill()
                raise RuntimeError(f"claude CLI timed out after {self.timeout}s") from exc
        text = _parse_cli_output(
            stdout_b.decode(errors="replace"),
            stderr_b.decode(errors="replace"),
            proc.returncode or 0,
        )
        message = AIMessage(content=text)
        return ChatResult(generations=[ChatGeneration(message=message)])

    def with_structured_output(
        self,
        schema: Any,
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable:
        """Return a runnable that coerces replies into *schema*.

        The CLI can't enforce a response schema, so we inject it into the
        prompt and validate the reply (with one repair retry).  Only Pydantic
        v2 model classes are supported — which is all SkillSpector uses.
        """
        if not (isinstance(schema, type) and hasattr(schema, "model_json_schema")):
            raise TypeError(
                "ChatClaudeCli.with_structured_output only supports Pydantic v2 "
                f"model classes; got {schema!r}"
            )
        return _StructuredClaudeCli(self, schema)


def _coerce_prompt(value: Any) -> str:
    """Best-effort render of a LangChain runnable input into prompt text."""
    if isinstance(value, str):
        return value
    if hasattr(value, "to_string"):
        return value.to_string()
    if isinstance(value, list):  # list[BaseMessage]
        return "\n\n".join(_message_text(getattr(m, "content", m)) for m in value)
    return str(value)


def _extract_json(text: str) -> Any:
    """Pull the first JSON object/array out of *text*, tolerating code fences."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        # Strip a leading ```json / ``` fence and the trailing fence.
        cleaned = cleaned.split("\n", 1)[-1] if "\n" in cleaned else cleaned
        if cleaned.endswith("```"):
            cleaned = cleaned[: -len("```")]
        cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Fall back to the outermost {...} or [...] span.
    starts = [i for i in (cleaned.find("{"), cleaned.find("[")) if i != -1]
    ends = [i for i in (cleaned.rfind("}"), cleaned.rfind("]")) if i != -1]
    if starts and ends:
        snippet = cleaned[min(starts) : max(ends) + 1]
        return json.loads(snippet)
    raise json.JSONDecodeError("No JSON object found in CLI response", cleaned, 0)


class _StructuredClaudeCli(Runnable):
    """Runnable wrapper that asks the CLI for JSON and validates it."""

    def __init__(self, llm: ChatClaudeCli, schema: type):
        self._llm = llm
        self._schema = schema

    def _augment(self, prompt: str) -> str:
        json_schema = json.dumps(self._schema.model_json_schema(), indent=2)
        return (
            f"{prompt}\n\n"
            "Respond with ONLY a single JSON value that conforms to the JSON "
            "Schema below. Do not include any prose, explanation, or Markdown "
            "code fences — output raw JSON only.\n\n"
            f"JSON Schema:\n{json_schema}"
        )

    def _repair(self, prompt: str, error: str) -> str:
        return (
            f"{self._augment(prompt)}\n\n"
            f"Your previous reply could not be parsed: {error}\n"
            "Return ONLY valid JSON matching the schema, nothing else."
        )

    def _validate(self, text: str) -> Any:
        return self._schema.model_validate(_extract_json(text))

    def invoke(self, input: Any, config: RunnableConfig | None = None, **kwargs: Any) -> Any:
        prompt = _coerce_prompt(input)
        text = str(self._llm.invoke(self._augment(prompt), config=config).content)
        try:
            return self._validate(text)
        except Exception as exc:  # noqa: BLE001 — one repair retry then re-raise
            logger.debug("Structured parse failed (%s); retrying once", exc)
            retry = str(self._llm.invoke(self._repair(prompt, str(exc)), config=config).content)
            return self._validate(retry)

    async def ainvoke(self, input: Any, config: RunnableConfig | None = None, **kwargs: Any) -> Any:
        prompt = _coerce_prompt(input)
        resp = await self._llm.ainvoke(self._augment(prompt), config=config)
        text = str(resp.content)
        try:
            return self._validate(text)
        except Exception as exc:  # noqa: BLE001 — one repair retry then re-raise
            logger.debug("Structured parse failed (%s); retrying once", exc)
            retry_resp = await self._llm.ainvoke(self._repair(prompt, str(exc)), config=config)
            return self._validate(str(retry_resp.content))
