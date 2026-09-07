"""Gemini API key 不可落在 URL 裡的守門測試——驗證 requests.post 實際收到的
(url, headers)，不是常數本身，這樣常數寫錯時測試才抓得到。
"""
import os
import sys

_REPO = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, _REPO)

from src import analyzer  # noqa: E402

DUMMY_KEY = "dummy-key-for-test"


class _FakeResponse:
    def __init__(self, json_body):
        self.status_code = 200
        self._json = json_body
        self.headers = {}
        self.text = ""

    def raise_for_status(self):
        pass

    def json(self):
        return self._json


def test_generate_embedding_key_in_header_not_url(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", DUMMY_KEY)
    captured = {}

    def fake_post(url, json=None, timeout=None, headers=None, **kwargs):
        captured["url"] = url
        captured["headers"] = headers
        return _FakeResponse({"embedding": {"values": [0.1, 0.2]}})

    monkeypatch.setattr(analyzer.requests, "post", fake_post)

    result = analyzer.generate_embedding("some text")

    assert result == [0.1, 0.2]
    assert "key=" not in captured["url"], "URL 不該含 key= 這個查詢參數"
    assert DUMMY_KEY not in captured["url"], "URL 不該含 API key 本體"
    assert captured["headers"]["x-goog-api-key"] == DUMMY_KEY


def test_call_gemini_key_in_header_not_url(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", DUMMY_KEY)
    captured = {}

    def fake_post(url, json=None, timeout=None, headers=None, **kwargs):
        captured["url"] = url
        captured["headers"] = headers
        body = {
            "candidates": [
                {"content": {"parts": [{"text": "ok"}]}, "finishReason": "STOP"}
            ]
        }
        return _FakeResponse(body)

    monkeypatch.setattr(analyzer.requests, "post", fake_post)

    result = analyzer._call_gemini("prompt", DUMMY_KEY)

    assert result == "ok"
    assert "key=" not in captured["url"], "URL 不該含 key= 這個查詢參數"
    assert DUMMY_KEY not in captured["url"], "URL 不該含 API key 本體"
    assert captured["headers"]["x-goog-api-key"] == DUMMY_KEY
