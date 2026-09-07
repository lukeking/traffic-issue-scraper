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


def test_call_gemini_sends_header_on_every_retry(monkeypatch):
    """header 建在 retry 迴圈外，所以只驗 200 那條路看不出重試有沒有帶它。
    第 481 行的 `logger.warning(..., e)` 會印出例外（含 URL），正是這個修補要防的。
    """
    monkeypatch.setenv("GEMINI_API_KEY", DUMMY_KEY)
    monkeypatch.setattr(analyzer.time, "sleep", lambda _s: None)
    calls = []

    class _Fail(_FakeResponse):
        def __init__(self):
            super().__init__({})
            self.status_code = 500

    def fake_post(url, json=None, timeout=None, headers=None, **kwargs):
        calls.append({"url": url, "headers": headers})
        if len(calls) < 3:
            return _Fail()
        return _FakeResponse(
            {"candidates": [{"content": {"parts": [{"text": "ok"}]}, "finishReason": "STOP"}]}
        )

    monkeypatch.setattr(analyzer.requests, "post", fake_post)

    assert analyzer._call_gemini("prompt", DUMMY_KEY, retries=5) == "ok"
    assert len(calls) == 3, f"應該重試到第 3 次才成功，實際 {len(calls)} 次"
    for i, c in enumerate(calls, 1):
        assert "key=" not in c["url"], f"第 {i} 次的 URL 不該含 key="
        assert c["headers"]["x-goog-api-key"] == DUMMY_KEY, f"第 {i} 次沒帶 header"
