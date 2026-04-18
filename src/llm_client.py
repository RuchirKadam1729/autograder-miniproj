"""
LLM client for communicating with an Ollama-backed language model.
Handles streaming NDJSON responses and JSON extraction with fallbacks.
"""

import json
import logging
import re
from typing import Any, Optional

import requests

from .config import LLMConfig

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Raised when the LLM request or response parsing fails."""


class LLMClient:
    """Stateless client for a single Ollama endpoint."""

    def __init__(self, cfg: LLMConfig) -> None:
        self._cfg = cfg

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, prompt: str) -> Optional[Any]:
        """
        Send *prompt* to the model and return the parsed JSON payload.

        Returns the parsed Python object on success, or ``None`` when the
        response cannot be parsed as JSON.

        Raises :class:`LLMError` for network / HTTP failures.
        """
        raw = self._stream_response(prompt)
        return self._parse_json(raw)

    def is_available(self) -> bool:
        """Smoke-test the LLM endpoint with a tiny grading prompt."""
        test_prompt = (
            'Grade this answer. Output ONLY a JSON array.\n'
            'Format: [{"point":"test","allocated_marks":5,'
            '"marks_awarded":3,"explanation":"ok"}]\n'
            'Student answer: test\nJSON array only:'
        )
        try:
            result = self.evaluate(test_prompt)
            return result is not None
        except LLMError as exc:
            logger.warning("LLM availability check failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _stream_response(self, prompt: str) -> str:
        """POST the prompt and accumulate streamed NDJSON into a string."""
        payload = {
            "model": self._cfg.model,
            "prompt": prompt,
            "max_tokens": self._cfg.max_tokens,
            "stream": self._cfg.stream,
        }
        headers = {"Content-Type": "application/json"}

        try:
            response = requests.post(
                self._cfg.url,
                json=payload,
                headers=headers,
                stream=True,
                timeout=self._cfg.timeout_seconds,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise LLMError(f"HTTP request failed: {exc}") from exc

        chunks: list[str] = []
        for raw_line in response.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
            try:
                chunk = json.loads(line)
                if isinstance(chunk, dict):
                    chunks.append(chunk.get("response", ""))
                    if chunk.get("done", False):
                        break
            except json.JSONDecodeError:
                continue

        text = "".join(chunks).strip()
        if not text:
            raise LLMError("Empty response received from LLM stream.")
        return text

    @staticmethod
    def _parse_json(text: str) -> Optional[Any]:
        """Try several strategies to coerce *text* into a Python object."""
        # Strip markdown fences
        cleaned = re.sub(r"```json\s*", "", text, flags=re.IGNORECASE)
        cleaned = re.sub(r"```\s*", "", cleaned).strip()

        # Strategy 1 — direct parse
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Strategy 2 — extract first JSON array
        match = re.search(r"\[\s*\{.*?\}\s*\]", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        # Strategy 3 — fix common encoding issues and trailing commas
        fixed = (
            cleaned
            .replace("\u201c", '"').replace("\u201d", '"')
            .replace("\u2018", "'").replace("\u2019", "'")
        )
        fixed = re.sub(r",(\s*[\]}])", r"\1", fixed)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

        logger.error(
            "JSON parse failed. First 200 chars: %s", cleaned[:200]
        )
        return None
