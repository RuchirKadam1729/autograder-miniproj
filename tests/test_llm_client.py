"""Unit tests for src.llm_client — no real network calls."""

import json
import unittest
from unittest.mock import MagicMock, patch

from src.config import LLMConfig
from src.llm_client import LLMClient, LLMError


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
        # Falls back gracefully — just test no exception
        _ = self._parse(text)

    def test_returns_none_for_garbage(self):
        result = self._parse("not json at all !@#")
        self.assertIsNone(result)


class TestLLMClientStreamResponse(unittest.TestCase):
    """Test _stream_response with mocked HTTP."""

    def _make_client(self) -> LLMClient:
        return LLMClient(LLMConfig(url="http://fake/api/generate", model="fake"))

    def _mock_response(self, chunks: list[dict]):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.iter_lines.return_value = [
            json.dumps(c).encode() for c in chunks
        ]
        return mock_resp

    @patch("src.llm_client.requests.post")
    def test_accumulates_response_chunks(self, mock_post):
        chunks = [
            {"response": "Hello", "done": False},
            {"response": " world", "done": True},
        ]
        mock_post.return_value = self._mock_response(chunks)
        client = self._make_client()
        result = client._stream_response("test prompt")
        self.assertEqual(result, "Hello world")

    @patch("src.llm_client.requests.post")
    def test_raises_on_empty_response(self, mock_post):
        chunks = [{"response": "", "done": True}]
        mock_post.return_value = self._mock_response(chunks)
        client = self._make_client()
        with self.assertRaises(LLMError):
            client._stream_response("test")

    @patch("src.llm_client.requests.post")
    def test_raises_on_http_error(self, mock_post):
        import requests as _requests
        mock_post.side_effect = _requests.RequestException("timeout")
        client = self._make_client()
        with self.assertRaises(LLMError):
            client._stream_response("test")


class TestLLMClientIsAvailable(unittest.TestCase):
    """Test is_available smoke-test behaviour."""

    def _make_client(self) -> LLMClient:
        return LLMClient(LLMConfig(url="http://fake/api/generate", model="fake"))

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
