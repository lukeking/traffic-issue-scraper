"""
`scripts/auto_kb.py` 的 Step 5（內聯修補）測試。

**為什麼是這幾條**：2026-08-31 有一個 bug 連著兩個 PR 都沒被攔下來——
Step 2 的「沒有未知術語 → `sys.exit(0)`」讓 Step 5 永遠到不了，而 Step 5 後來長出
兩個獨立於 Gemini 的職責（修補全 KB 的 jp→tw、拆 `IGNORED_MARKERS` 的括號），
那兩者恰好會把未知集清空，於是自己觸發早退、把自己關在門外。

那是一個**控制流** bug，不是邏輯 bug。當時的驗證方式是分別乾跑 Step 2 與 Step 5
的邏輯，兩邊都「正確」——但沒有任何東西模擬它們之間的 `sys.exit`。
所以這裡刻意跑**真正的 `main()`**，用假 client 攔住寫入，而不是重寫一份邏輯：
重寫的那份不會有早退，於是永遠測不到這個 bug。

2026-09-05 擴充（BACKLOG #12）：同一個 harness 補上 **Step 4（KB 寫入分支）** 與
**`call_gemini` 的回應解析**——那兩塊在 09-04 逐行查證時是零覆蓋（`call_gemini` 在每條
測試裡都被 monkeypatch 掉，Step 4 的 `db.inserted` 有被收集但沒有任何測試讀它）。
Step 4 的測試一律走 `run_main`，理由同上：Step 2 的早退就在它前面。
"""
import json
import os
import sys
import types

import pytest

_REPO = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "scripts"))

import auto_kb  # noqa: E402


# ── 假 supabase client：只支援 auto_kb 實際用到的那幾條鏈 ──────────────────

class _Query:
    def __init__(self, table, op, payload=None):
        self.table, self.op, self.payload = table, op, payload
        self.filters = {}

    def eq(self, col, val):
        self.filters[col] = val
        return self

    def execute(self):
        return types.SimpleNamespace(data=self.table._run(self))


class _Table:
    def __init__(self, db, name):
        self.db, self.name = db, name

    def select(self, _cols):
        return _Query(self, "select")

    def insert(self, rows):
        return _Query(self, "insert", rows)

    def update(self, patch):
        return _Query(self, "update", patch)

    def _run(self, q):
        if self.name == "knowledge_base":
            if q.op == "select":
                return [dict(r) for r in self.db.kb]
            if q.op == "insert":
                rows = q.payload if isinstance(q.payload, list) else [q.payload]
                for r in rows:
                    if r.get("jp_term") in self.db.insert_fails:
                        raise RuntimeError(f"模擬寫入失敗：{r.get('jp_term')}")
                self.db.kb.extend(rows)
                self.db.inserted.extend(rows)
                return rows
        if self.name == "articles":
            if q.op == "select":
                return [dict(a) for a in self.db.articles]
            if q.op == "update":
                aid = q.filters.get("id")
                for a in self.db.articles:
                    if a["id"] == aid:
                        a.update(q.payload)
                self.db.updates.append((aid, q.payload))
                return []
        raise AssertionError(f"未預期的呼叫：{self.name}.{q.op}")


class _FakeDB:
    def __init__(self, kb, articles, insert_fails=None):
        self.kb, self.articles = kb, articles
        self.updates, self.inserted = [], []
        # 指定的 jp_term 寫入時拋例外，用來咬 Step 4「寫入失敗要吞掉並繼續」那條分支。
        self.insert_fails = insert_fails or set()

    def table(self, name):
        return _Table(self, name)


@pytest.fixture
def run_main(monkeypatch):
    """跑真正的 main()，回傳假 DB 供斷言。gemini_calls 記錄它有沒有呼叫 Gemini。

    `gemini_entries` ＝ 假 `call_gemini` 要回傳的東西（預設空陣列＝原本的行為）；
    `insert_fails` ＝ 一組 jp_term，假 DB 寫到它們時會拋例外。
    """
    def _run(kb, articles, gemini_entries=None, insert_fails=None):
        db = _FakeDB(kb, articles, insert_fails)
        monkeypatch.setitem(
            sys.modules, "supabase",
            types.SimpleNamespace(create_client=lambda url, key: db),
        )
        monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
        monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "fake-not-a-real-key")
        monkeypatch.setenv("GEMINI_API_KEY", "fake-not-a-real-key")

        calls = []
        monkeypatch.setattr(
            auto_kb, "call_gemini",
            lambda terms, *a, **k: (calls.append(sorted(terms)) or list(gemini_entries or [])),
        )
        # main() 正常路徑是直接回傳；早退只發生在錯誤或無事可做的分支，
        # 且依 docstring「Always exits 0」一律是 0。兩種都接受，非零則失敗。
        try:
            auto_kb.main()
        except SystemExit as exc:
            assert exc.code in (0, None), f"main() 以非零碼結束：{exc.code}"
        db.gemini_calls = calls
        return db
    return _run


def _article(aid, text):
    return {"id": aid, "analysis": {"summary": text}}


def _summary(db, aid):
    return next(a for a in db.articles if a["id"] == aid)["analysis"]["summary"]


# ── 迴歸：這條就是 2026-08-31 漏掉的那個案例 ──────────────────────────────

def test_known_term_patched_with_no_unknowns(run_main):
    kb = [{"jp_term": "ララ", "tw_term": "拉拉菲爾族"}]
    arts = [_article(1, "自己的 [[ララ]] 角色")]
    db = run_main(kb, arts)

    assert db.gemini_calls == [], "沒有未知術語就不該呼叫 Gemini"
    assert _summary(db, 1) == "自己的 拉拉菲爾族 角色", "已知詞必須被修補"
    assert db.updates, "文章沒有被寫回"


def test_ignored_marker_loses_brackets_with_no_unknowns(run_main):
    """`IGNORED_MARKERS` 的詞拆括號留原文，且同樣不該被早退擋掉。"""
    arts = [_article(2, "阿莉澤與 [[spiky boy]] 將作為夥伴")]
    db = run_main([], arts)

    assert db.gemini_calls == []
    assert _summary(db, 2) == "阿莉澤與 spiky boy 將作為夥伴"


def test_unknown_term_still_reaches_gemini(run_main):
    """未知詞照舊送 Gemini；修法不能把原本的主線關掉。"""
    arts = [_article(3, "任務 [[某個沒人知道的詞]] 結束後")]
    db = run_main([], arts)

    assert db.gemini_calls == [["某個沒人知道的詞"]]
    assert "[[某個沒人知道的詞]]" in _summary(db, 3), "Gemini 沒解出來就該保持原樣"


def test_ignored_marker_not_sent_to_gemini(run_main):
    """忽略名單的詞不該浪費 Gemini 呼叫——它每輪都會被婉拒。"""
    arts = [_article(4, "[[spiky boy]] 與 [[另一個未知詞]]")]
    db = run_main([], arts)

    assert db.gemini_calls == [["另一個未知詞"]]


def test_no_markers_at_all_is_a_noop(run_main):
    """完全沒有標記時不寫入任何東西（別為了修早退而變成每輪全表重寫）。"""
    db = run_main([{"jp_term": "ララ", "tw_term": "拉拉菲爾族"}],
                  [_article(5, "一篇沒有任何標記的摘要")])

    assert db.updates == []
    assert db.gemini_calls == []


# ── 繁中正規化（TW_NORMALISATION）：陸服詞 → 官方繁中詞 ────────────────────
#
# 判準是兩條：陸服詞換成官方繁中；縮寫不展開。所以這一組有兩種測試——
# 一種證明**該換的有換**（優先序贏過 KB），一種證明**不該換的沒被換**（守門員）。
#
# KB fixture 都刻意造出撞號：同一個字串既是某列的 `jp_term` 也是另一列的 `tw_term`，
# 讓 `known_jp` / `known_tw` 都有機會贏。

def test_tw_normalisation_beats_known_tw(run_main):
    """`地下城`（陸服詞）→ `迷宮`，即使 KB 有一列 jp→`地下城` 讓 known_tw 佔著同一個鍵。"""
    kb = [{"jp_term": "ダンジョン", "tw_term": "地下城"}]
    arts = [_article(10, "本次更新追加了 [[地下城]] 挑戰")]
    db = run_main(kb, arts)

    assert db.gemini_calls == [], "純正規化的一輪不該呼叫 Gemini"
    assert _summary(db, 10) == "本次更新追加了 迷宮 挑戰"


def test_abbreviation_is_not_expanded_fc(run_main):
    """`FC` **不展開**成官方全稱 `公會`——規則 2，縮寫對玩家往往比全稱好讀。

    KB fixture 照 prod 的實際形狀擺：id=122 `フリーカンパニー→FC` 讓 `known_tw`
    給出 `FC→FC`，id=147 `FC→公會`（2026-09-03 已更正為官方繁中）讓 `known_jp`
    給出 `FC→公會`。`known_tw` 贏，所以標記只拆括號、保留 `FC`。

    這條是守門員：有人日後把 `FC` 加進 TW_NORMALISATION，或翻轉合併順序，它會紅。
    """
    kb = [
        {"jp_term": "フリーカンパニー", "tw_term": "FC"},
        {"jp_term": "FC", "tw_term": "公會"},
    ]
    arts = [_article(11, "加入 [[FC]] 之後")]
    db = run_main(kb, arts)

    assert db.gemini_calls == []
    summary = _summary(db, 11)
    assert summary == "加入 FC 之後"
    assert "公會" not in summary, "縮寫不該被展開成官方全稱"


def test_abbreviation_is_not_expanded_cf(run_main):
    """`CF` 同理不展開成 `任務搜尋器`。"""
    kb = [
        {"jp_term": "コンテンツファインダー", "tw_term": "CF"},
        {"jp_term": "CF", "tw_term": "任務搜尋器"},
    ]
    arts = [_article(12, "請用 [[CF]] 排隊")]
    db = run_main(kb, arts)

    assert db.gemini_calls == []
    summary = _summary(db, 12)
    assert summary == "請用 CF 排隊"
    assert "任務搜尋器" not in summary, "縮寫不該被展開成官方全稱"


def test_normalisation_table_holds_only_verified_locale_swaps(run_main):
    """表裡只該有「陸服詞→官方繁中」，不該混進縮寫展開。

    直接斷言內容而非行為：上面兩條守的是**結果**，這條守的是**規則本身**，
    這樣「為什麼 FC 不在裡面」不需要靠讀註解才知道。
    """
    assert auto_kb.TW_NORMALISATION == {"地下城": "迷宮"}


def test_excluded_term_keeps_original_word(run_main):
    """`極本` 刻意不在正規化表裡：官方繁中是前綴構詞（「極 某某殲滅戰」），沒有單詞對應。

    所以它只拆括號、保留原文，**不可以**變成 KB 那列的 `極神`——那是陸服詞。
    這條是**排除**的守門員：有人日後「順手補完」表格時它會紅。
    """
    kb = [
        {"jp_term": "極討滅戦", "tw_term": "極本"},
        {"jp_term": "極本", "tw_term": "極神"},
    ]
    arts = [_article(13, "打了幾場 [[極本]] 之後")]
    db = run_main(kb, arts)

    assert db.gemini_calls == []
    summary = _summary(db, 13)
    assert summary == "打了幾場 極本 之後"
    assert "極神" not in summary, "極神 是陸服詞，不該被正規化表帶進來"


# ── Step 4：Gemini 回應 → knowledge_base 寫入 ──────────────────────────────
#
# 09-04 逐行查證時這塊是零覆蓋：假 client 一直有收集 `db.inserted`，但沒有任何測試
# 讀它，所以三條跳過／容錯分支都沒被咬到。以下四條對應 Step 4 的四條路徑。

def test_step4_inserts_valid_entry_and_patches_article(run_main):
    """正常路徑：寫進 KB，且**同一輪**就把文章裡的標記換掉（`newly_added` 餵進 Step 5）。"""
    arts = [_article(20, "討伐 [[絶バハムート討滅戦]] 的隊伍")]
    db = run_main([], arts, gemini_entries=[{
        "jp_term": "絶バハムート討滅戦", "tw_term": "絕巴哈姆特討滅戰",
        "en_term": "UCoB", "category": "副本", "notes": "4.11（2018）",
    }])

    assert db.gemini_calls == [["絶バハムート討滅戦"]]
    assert db.inserted == [{
        "jp_term": "絶バハムート討滅戦", "tw_term": "絕巴哈姆特討滅戰",
        "en_term": "UCoB", "category": "副本", "notes": "4.11（2018）",
        "auto_generated": True,
    }], "寫入的欄位要逐字相符，且 auto_generated 必須為 True"
    assert _summary(db, 20) == "討伐 絕巴哈姆特討滅戰 的隊伍"


def test_step4_fills_optional_fields_with_empty_string(run_main):
    """Gemini 只回必要欄位時，`en_term`／`category`／`notes` 補空字串（`or ""`）而非 None。"""
    arts = [_article(21, "打完 [[オメガ]] 之後")]
    db = run_main([], arts, gemini_entries=[{"jp_term": "オメガ", "tw_term": "歐米茄"}])

    assert db.inserted == [{
        "jp_term": "オメガ", "tw_term": "歐米茄",
        "en_term": "", "category": "", "notes": "", "auto_generated": True,
    }]


def test_step4_skips_entry_missing_a_required_field(run_main):
    """缺 `jp_term` 或 `tw_term` 的項目跳過——不寫 KB，也不拿它去改文章。"""
    arts = [_article(22, "地名 [[グリダニア]] 與 [[ウルダハ]]")]
    db = run_main([], arts, gemini_entries=[
        {"jp_term": "グリダニア", "tw_term": ""},   # tw 空字串
        {"jp_term": "", "tw_term": "烏爾達哈"},      # jp 空字串
        {"tw_term": "只有繁中"},                     # jp 整個缺席
    ])

    assert db.inserted == []
    assert _summary(db, 22) == "地名 [[グリダニア]] 與 [[ウルダハ]]", "跳過的項目不該改到文章"
    assert db.updates == []


def test_step4_skips_term_already_in_kb(run_main):
    """Gemini 回了一個 KB 已有的 `jp_term` → 跳過，既有譯名不被覆蓋。

    ⚠️ 要走到 Step 4，那個詞得先被送出去——所以文章裡放的是**另一個**未知詞，
    由 Gemini「順便」多回一筆已存在的。把已存在的詞直接放進文章會被 Step 2 先濾掉，
    根本到不了 Step 4，那樣測到的是別條分支。
    """
    kb = [{"jp_term": "エオルゼア", "tw_term": "艾歐澤亞"}]
    arts = [_article(23, "從 [[某個未知地名]] 出發")]
    db = run_main(kb, arts, gemini_entries=[
        {"jp_term": "エオルゼア", "tw_term": "伊歐澤亞"},   # 已存在，且譯名不同
        {"jp_term": "某個未知地名", "tw_term": "某個地名"},
    ])

    assert [r["jp_term"] for r in db.inserted] == ["某個未知地名"]
    assert _summary(db, 23) == "從 某個地名 出發"


def test_step4_insert_failure_is_swallowed_and_loop_continues(run_main):
    """某一筆寫入拋例外 → 記 warning 並繼續下一筆（Auto-KB 全程 non-blocking）。

    失敗那筆**不可以**進 `newly_added`，否則文章會被改成一個 KB 裡並不存在的譯名。
    """
    arts = [_article(24, "[[前一個詞]] 與 [[後一個詞]]")]
    db = run_main([], arts, gemini_entries=[
        {"jp_term": "前一個詞", "tw_term": "前譯"},
        {"jp_term": "後一個詞", "tw_term": "後譯"},
    ], insert_fails={"前一個詞"})

    assert [r["jp_term"] for r in db.inserted] == ["後一個詞"], "第二筆必須照樣寫入"
    assert _summary(db, 24) == "[[前一個詞]] 與 後譯", "寫入失敗的詞不該被替換掉"


# ── `call_gemini`：回應解析與重試 ──────────────────────────────────────────
#
# `call_gemini`（`scripts/auto_kb.py:132`）在上面每一條測試裡都被 monkeypatch 掉，
# 本體一行都沒被執行過。這一組**不測 HTTP**，只把 `requests` 換掉（替換邊界，不是
# 重寫邏輯），測「拿到回應之後怎麼解析」與「壞掉時退回什麼」——後者承重，因為它的
# 失敗模式是**回傳空陣列繼續跑**（non-blocking），不是拋例外，所以壞掉是沉默的。

class _FakeResponse:
    def __init__(self, text=None, status_code=200):
        self.status_code = status_code
        self._text = text
        self.raise_for_status_called = False

    def raise_for_status(self):
        self.raise_for_status_called = True
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return {"candidates": [{"content": {"parts": [{"text": self._text}]}}]}


@pytest.fixture
def fake_requests(monkeypatch):
    """把 `requests` 換成假的。`queue` 依序供應回應，`posts` 記錄送出去的東西。"""
    calls = types.SimpleNamespace(posts=[], queue=[])

    def _post(url, json=None, timeout=None, headers=None):
        calls.posts.append({"url": url, "payload": json, "headers": headers})
        return calls.queue.pop(0)

    monkeypatch.setitem(sys.modules, "requests", types.SimpleNamespace(post=_post))
    # 重試路徑會 sleep 1 秒與 2 秒；測試不該為此等 3 秒。
    monkeypatch.setattr(auto_kb.time, "sleep", lambda _s: None)
    return calls


def test_call_gemini_parses_plain_json(fake_requests):
    fake_requests.queue.append(_FakeResponse(
        '[{"jp_term": "暁月のフィナーレ", "tw_term": "曉月", "en_term": "Endwalker", '
        '"category": "資料片", "notes": "6.0（2021）"}]'
    ))

    out = auto_kb.call_gemini(["暁月のフィナーレ"], "fake-not-a-real-key", "gemini-x-flash")

    assert out == [{"jp_term": "暁月のフィナーレ", "tw_term": "曉月", "en_term": "Endwalker",
                    "category": "資料片", "notes": "6.0（2021）"}]
    # prompt 組裝的最小斷言：術語真的有進到送出去的 payload，模型名真的有進 URL。
    assert "暁月のフィナーレ" in fake_requests.posts[0]["payload"]["contents"][0]["parts"][0]["text"]
    assert "gemini-x-flash" in fake_requests.posts[0]["url"]


def test_call_gemini_strips_markdown_fences(fake_requests):
    """Gemini 常把 JSON 包在 ```json 圍欄裡；不拆就 `json.loads` 失敗、整輪退回空陣列。"""
    fake_requests.queue.append(
        _FakeResponse('```json\n[{"jp_term": "極討滅戦", "tw_term": "極"}]\n```')
    )

    out = auto_kb.call_gemini(["極討滅戦"], "fake-not-a-real-key", "gemini-x-flash")

    assert out == [{"jp_term": "極討滅戦", "tw_term": "極"}]


def test_call_gemini_returns_empty_list_on_unparseable_response(fake_requests):
    """壞掉的 JSON **不可以**往上拋——Auto-KB 是 non-blocking 的排程工作。
    三次都壞 → 回空陣列，於是 Step 4 什麼都不寫、Step 5 照樣跑。"""
    for _ in range(3):
        fake_requests.queue.append(_FakeResponse("這不是 JSON"))

    out = auto_kb.call_gemini(["某詞"], "fake-not-a-real-key", "gemini-x-flash")

    assert out == []
    assert len(fake_requests.posts) == 3, "解析失敗要重試滿三次"


def test_call_gemini_retries_after_server_error(fake_requests):
    """5xx 走專屬的 `continue` 重試，**不經** `raise_for_status`；下一次成功就回那次的結果。

    ⚠️ 只斷言「重試了兩次」是分不出來的：泛用 `except` 那條路徑同樣會重試並成功。
    能把兩者分開的只有一件事——5xx 那個回應的 `raise_for_status` 有沒有被呼叫。
    """
    server_error = _FakeResponse(status_code=503)
    fake_requests.queue.append(server_error)
    fake_requests.queue.append(_FakeResponse('[{"jp_term": "オメガ", "tw_term": "歐米茄"}]'))

    out = auto_kb.call_gemini(["オメガ"], "fake-not-a-real-key", "gemini-x-flash")

    assert out == [{"jp_term": "オメガ", "tw_term": "歐米茄"}]
    assert len(fake_requests.posts) == 2
    assert not server_error.raise_for_status_called, "5xx 應在 raise_for_status 之前就 continue"


def test_call_gemini_key_in_header_not_url(fake_requests):
    """key 若留在 URL 查詢字串，第 161 行的 `logger.warning(..., exc)` 會把它印進 log
    ——例外訊息含完整 URL。所以斷言的是 requests 實際收到的 (url, headers)。
    """
    fake_requests.queue.append(_FakeResponse('[{"jp_term": "オメガ", "tw_term": "歐米茄"}]'))

    auto_kb.call_gemini(["オメガ"], "dummy-key-for-test", "gemini-x-flash")

    sent = fake_requests.posts[0]
    assert "key=" not in sent["url"], "URL 不該含 key= 這個查詢參數"
    assert "dummy-key-for-test" not in sent["url"], "URL 不該含 API key 本體"
    assert sent["headers"]["x-goog-api-key"] == "dummy-key-for-test"


def test_call_gemini_sends_header_on_every_retry(fake_requests):
    """headers 建在 for 迴圈外，只驗第一次看不出重試有沒有帶它——
    而重試那條路徑正是會把例外（含 URL）寫進 log 的那條。
    """
    fake_requests.queue.append(_FakeResponse(status_code=503))
    fake_requests.queue.append(_FakeResponse(status_code=503))
    fake_requests.queue.append(_FakeResponse('[{"jp_term": "オメガ", "tw_term": "歐米茄"}]'))

    auto_kb.call_gemini(["オメガ"], "dummy-key-for-test", "gemini-x-flash")

    assert len(fake_requests.posts) == 3, f"應重試到第 3 次，實際 {len(fake_requests.posts)} 次"
    for i, sent in enumerate(fake_requests.posts, 1):
        assert "key=" not in sent["url"], f"第 {i} 次的 URL 不該含 key="
        assert sent["headers"]["x-goog-api-key"] == "dummy-key-for-test", f"第 {i} 次沒帶 header"
