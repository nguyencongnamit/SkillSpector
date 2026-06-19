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

"""Claude-CLI provider — Claude models via the local ``claude`` subscription.

Selected with ``SKILLSPECTOR_PROVIDER=claude_cli``.  Unlike
:class:`AnthropicProvider`, this provider needs **no API key**: it drives the
host ``claude`` CLI, which authenticates with a Claude Max/Pro subscription
(macOS Keychain) or a ``CLAUDE_CODE_OAUTH_TOKEN``.  See
:mod:`skillspector.providers.claude_cli.chat_model` for the trade-offs.

This is intended for interactive / local runs on a machine where ``claude`` is
logged in — NOT for unattended CI, which has no subscription session to borrow
and should keep using an API-key provider.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel

from skillspector.providers import registry

from .chat_model import ChatClaudeCli

REGISTRY_PATH = str(Path(__file__).with_name("model_registry.yaml"))


class ClaudeCliProvider:
    """Subscription-backed provider that shells out to the ``claude`` CLI."""

    DEFAULT_MODEL = "claude-opus-4-6"
    SLOT_DEFAULTS: dict[str, str] = {
        # Cheaper/faster model for the high-volume meta-analyzer pass.
        "meta_analyzer": "claude-sonnet-4-6",
    }

    def _cli_path(self) -> str:
        return os.environ.get("SKILLSPECTOR_CLAUDE_CLI", "claude")

    def resolve_credentials(self) -> tuple[str, str | None] | None:
        """Signal availability when the ``claude`` binary is on PATH.

        There is no API key; the CLI owns authentication.  We return a
        sentinel ``(token, None)`` so SkillSpector's ``is_llm_available``
        check passes.  When ``claude`` is not installed we return ``None`` so
        callers can fall back to an API-key provider.
        """
        if shutil.which(self._cli_path()) is None:
            return None
        return "claude-cli-subscription", None

    def create_chat_model(
        self,
        model: str,
        *,
        max_tokens: int,
        timeout: float | None = 120,
    ) -> BaseChatModel | None:
        """Create a :class:`ChatClaudeCli` bound to *model*."""
        if shutil.which(self._cli_path()) is None:
            return None
        env_timeout = os.environ.get("SKILLSPECTOR_CLAUDE_TIMEOUT", "").strip()
        resolved_timeout: float | None
        try:
            resolved_timeout = float(env_timeout) if env_timeout else max(timeout or 0, 600)
        except ValueError:
            resolved_timeout = 600.0
        return ChatClaudeCli(
            claude_model=model,
            cli_path=self._cli_path(),
            timeout=resolved_timeout,
            max_output_tokens=max_tokens,
        )

    def get_context_length(self, model: str) -> int | None:
        return registry.lookup_context_length(REGISTRY_PATH, model)

    def get_max_output_tokens(self, model: str) -> int | None:
        return registry.lookup_max_output_tokens(REGISTRY_PATH, model)

    def resolve_model(self, slot: str = "default") -> str:
        """Resolve model: ``SKILLSPECTOR_MODEL`` env > slot default > ``DEFAULT_MODEL``."""
        user_input = os.environ.get("SKILLSPECTOR_MODEL", "").strip()
        return user_input or self.SLOT_DEFAULTS.get(slot, "") or self.DEFAULT_MODEL
