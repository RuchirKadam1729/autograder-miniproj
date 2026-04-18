"""
LLM client backed by the Groq API (OpenAI-compatible).
Replaces the old Ollama streaming client — no server to keep alive.
"""

import json
import logging
import re
from typing import Any, Optional

from groq import Groq, APIError, APIConnectionError, RateLimitError

from .config import LLMConfig

logger = logging.getLogger(__name__)

GROQ_MODELS = [
    "llama-3.3-70b-versatile",   # best quality, generous free tier
    "llama-3.1-8b-instant",      # fastest
    "mixtral-8x7b-32768",        # long context
    "gemma2-9b-it",
]


class LLMError(Exception):
    """Raised when the LLM request or response parsing fails."""


class LLMClient:
    """Thin wrapper around the Groq chat-completions API."""

    def __init__(self, cfg: LLMConfig) -> None:
        if not cfg.api_key:
            raise LLMError(
                "GROQ_API_KEY is not set. Get one free at https://console.groq.com"
            )
        self._cfg = cfg
        self._client = Groq(api_key=cfg.api_key)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, prompt: str) -> Optional[Any]:
        """
        Send *prompt* to Groq and return the parsed JSON payload.
        Returns the parsed Python object, or None if parsing fails.
        Raises LLMError for API / network failures.
        """
        raw = self._call(prompt)
        return self._parse_json(raw)

    def is_available(self) -> bool:
        """Smoke-test the API key with a tiny prompt."""
        test_prompt = (
            "Output ONLY this JSON array, nothing else:\n"
            '[{"point":"test","allocated_marks":5,"marks_awarded":3,"explanation":"ok"}]'
        )
        try:
            result = self.evaluate(test_prompt)
            return isinstance(result, list) and len(result) > 0
        except LLMError as exc:
            logger.warning("Groq availability check failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _call(self, prompt: str) -> str:
        """Make a single chat-completions request and return the text."""
        try:
            completion = self._client.chat.completions.create(
                model=self._cfg.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self._cfg.max_tokens,
                temperature=self._cfg.temperature,
            )
            return completion.choices[0].message.content or ""
        except RateLimitError as exc:
            raise LLMError(f"Groq rate limit hit: {exc}") from exc
        except APIConnectionError as exc:
            raise LLMError(f"Groq connection error: {exc}") from exc
        except APIError as exc:
            raise LLMError(f"Groq API error ({exc.status_code}): {exc.message}") from exc

    @staticmethod
    def _parse_json(text: str) -> Optional[Any]:
        """Try several strategies to coerce text into a Python object."""
        cleaned = re.sub(r"```json\s*", "", text, flags=re.IGNORECASE)
        cleaned = re.sub(r"```\s*", "", cleaned).strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        match = re.search(r"\[\s*\{.*?\}\s*\]", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

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

        logger.error("JSON parse failed. First 200 chars: %s", cleaned[:200])
        return None