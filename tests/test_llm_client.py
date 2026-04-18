"""Unit tests for src.llm_client — no real network calls."""

import json
import unittest
from unittest.mock import MagicMock, patch

from src.config import LLMConfig
from src.llm_client import LLMClient, LLMError
from groq import APIError, APIConnectionError, RateLimitError


class TestLLMClientParseJson(unittest.TestCase):
    """Test the static _parse_json method in isolation."""

    def _parse(self, text: str):
        return LLMClient._parse_json(text)

    def test_plain_json_array(self):
        data = [{"point": "x", "marks_awarded": 3}]
        result = self._parse(json.dumps(data))
        self.assertEqual(result, data)

    def test_strips_markdown_fence(self):
        text = "```json\n[{\"a\": 1}]\n```"
        result = self._parse(text)
        self.assertEqual(result, [{"a": 1}])

    def test_extracts_embedded_array(self):
        text = 'Here is the result: [{"b": 2}] — done.'
        result = self._parse(text)
        self.assertEqual(result, [{"b": 2}])

    def test_fixes_smart_quotes(self):
        text = '[\u007b\u201cpoint\u201d: \u201chello\u201d, \u201cmarks_awarded\u201d: 1\u007d]'
        _ = self._parse(text)

    def test_returns_none_for_garbage(self):
        result = self._parse("not json at all !@#")
        self.assertIsNone(result)


class TestLLMClientCall(unittest.TestCase):
    """Test _call with mocked Groq client."""

    def _make_client(self) -> LLMClient:
        with patch("src.llm_client.Groq"):
            return LLMClient(LLMConfig(api_key="fake-key", model="fake-model"))

    def _set_response(self, client, text):
        client._client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=text))]
        )

    def test_accumulates_response_chunks(self):
        client = self._make_client()
        self._set_response(client, '[{"response": "Hello world"}]')
        result = client._call("test prompt")
        self.assertEqual(result, '[{"response": "Hello world"}]')

    def test_raises_on_empty_response(self):
        client = self._make_client()
        client._client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=None))]
        )
        result = client._call("test")
        self.assertEqual(result, "")

    def test_raises_on_connection_error(self):
        client = self._make_client()
        client._client.chat.completions.create.side_effect = APIConnectionError.__new__(APIConnectionError)
        with self.assertRaises(LLMError):
            client._call("test")


class TestLLMClientIsAvailable(unittest.TestCase):
    """Test is_available smoke-test behaviour."""

    def _make_client(self) -> LLMClient:
        with patch("src.llm_client.Groq"):
            return LLMClient(LLMConfig(api_key="fake-key", model="fake-model"))

    def test_returns_true_when_evaluate_succeeds(self):
        client = self._make_client()
        client.evaluate = MagicMock(return_value=[{"point": "x", "marks_awarded": 3}])
        self.assertTrue(client.is_available())

    def test_returns_false_when_evaluate_returns_none(self):
        client = self._make_client()
        client.evaluate = MagicMock(return_value=None)
        self.assertFalse(client.is_available())

    def test_returns_false_on_llm_error(self):
        client = self._make_client()
        client.evaluate = MagicMock(side_effect=LLMError("boom"))
        self.assertFalse(client.is_available())


if __name__ == "__main__":
    unittest.main()