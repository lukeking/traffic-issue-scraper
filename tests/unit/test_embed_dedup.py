"""`src/analyzer.embed_dedup` 的測試。

**為什麼到現在才有**：它是評分路徑上唯一會打外部服務的函式（BACKLOG #4 稱之為
「唯一的雜質」），所以離線測不了，於是一直沒有測試——而另外四個純函式
（`cluster_traffic_articles` / `score_topic_buckets` / `select_hot_topics_with_novelty` /
`select_digest_pool`）早就各有零依賴的 unit 測試。**不純與沒測試落在同一個函式上，
不是巧合。** 把 `generate_embedding` 抽成 `attach_embeddings` 之後，這些測試不需要
monkeypatch 任何網路呼叫。

向量刻意用二維：`_cosine_similarity` 不在乎維度，而二維的餘弦值可以手算驗證，
不必依賴某個 768 維 fixture 的不透明數字。
"""
import logging
import os
import sys

_REPO = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, _REPO)

from src.analyzer import embed_dedup  # noqa: E402

# cos([1,0], [0.9,0.4]) = 0.9/√0.97 ≈ 0.914 → 高於 0.88
# cos([1,0], [0.8,0.6]) = 0.8/1.0    = 0.800 → 低於 0.88
NEAR = [0.9, 0.4]
FAR = [0.8, 0.6]
BASE = [1.0, 0.0]


def _art(aid, emb, summary="摘要"):
    return {"id": aid, "title": f"標題{aid}", "summary": summary, "embedding": emb}


def test_candidate_already_in_buffer_is_dropped():
    kept = embed_dedup([_art(1, BASE)], [_art(99, NEAR)], threshold=0.88)

    assert kept == [], "與本週 buffer 相似的候選要被丟掉"


def test_below_threshold_is_kept():
    kept = embed_dedup([_art(1, BASE)], [_art(99, FAR)], threshold=0.88)

    assert [a["id"] for a in kept] == [1], "0.80 低於門檻，不該被當成同一則"


def test_batch_dedup_keeps_the_longer_summary():
    """同一批裡的重複保留 summary 較長的那篇——短的那篇往往是標題回音。"""
    short = _art(1, BASE, summary="短")
    long_ = _art(2, NEAR, summary="這是一段明顯比較長的正文" * 5)

    kept = embed_dedup([short, long_], [], threshold=0.88)

    assert [a["id"] for a in kept] == [2]


def test_batch_dedup_keeps_first_when_candidate_is_not_longer():
    shorter_second = _art(2, NEAR, summary="短")
    kept = embed_dedup([_art(1, BASE, summary="比較長的正文" * 5), shorter_second], [], threshold=0.88)

    assert [a["id"] for a in kept] == [1]


def test_articles_without_a_vector_are_kept_not_dropped():
    """生成失敗（值為 None）不可以變成「被去重」——那會靜靜地丟掉文章。"""
    kept = embed_dedup([_art(1, None), _art(2, BASE)], [_art(99, NEAR)], threshold=0.88)

    assert [a["id"] for a in kept] == [1], "無向量者保留；有向量且命中 buffer 者丟掉"


def test_pgvector_string_embeddings_are_parsed():
    """DB 讀回來的 embedding 是 pgvector 的字串形式 '[x,y]'，不是 list。"""
    kept = embed_dedup([_art(1, "[1.0,0.0]")], [_art(99, "[0.9,0.4]")], threshold=0.88)

    assert kept == [], "字串向量沒被解析的話這裡會留下來"


def test_empty_candidates_is_a_noop():
    assert embed_dedup([], [_art(99, BASE)], threshold=0.88) == []


# ── 這條守的是 2026-09-05 這次重構本身 ────────────────────────────────────

def test_missing_embedding_key_warns_about_the_missing_wiring(caplog):
    """把 `generate_embedding` 抽出去之後，呼叫端漏喊 `attach_embeddings()` 的話
    這批就不會被去重——而**不去重是沉默的**（沒有錯誤、只是重複文章照樣進庫）。

    所以缺「鍵」必須發出一個**指名那個函式**的警告，且與「生成失敗（值為 None）」
    分開：兩者後果相同、成因不同，混在一起會讓接線錯誤看起來像 API 故障。
    """
    with caplog.at_level(logging.WARNING):
        kept = embed_dedup([{"id": 1, "title": "無鍵"}], [_art(99, BASE)], threshold=0.88)

    assert [a["id"] for a in kept] == [1]
    assert any("attach_embeddings" in r.getMessage() for r in caplog.records), \
        "警告必須指名 attach_embeddings，否則接線漏掉時查不出來"


def test_generation_failure_does_not_mention_the_wiring(caplog):
    """反面：值是 None（API 掛了）時，不該叫人去查 `attach_embeddings` 的接線。"""
    with caplog.at_level(logging.WARNING):
        embed_dedup([_art(1, None)], [], threshold=0.88)

    joined = " ".join(r.getMessage() for r in caplog.records)
    assert "attach_embeddings" not in joined


def test_generation_failure_is_counted_out_loud(caplog):
    """生成失敗必須有一行帶筆數的彙總警告。上一條只斷言不要說什麼，
    所以「什麼都不說」對它是綠的；這條斷言要說什麼。緣由見 BACKLOG #13。
    """
    with caplog.at_level(logging.WARNING):
        kept = embed_dedup(
            [_art(1, None), _art(2, None), _art(3, BASE)], [], threshold=0.88
        )

    assert sorted(a["id"] for a in kept) == [1, 2, 3], "無向量的文章要保留，不是丟掉"
    warns = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("2/3" in m for m in warns), \
        f"要有一行帶「2/3」的彙總警告，實際只有：{warns}"


def test_missing_key_is_not_double_counted_as_generation_failure(caplog):
    """缺鍵的那批解析後值也是 None，不可重複算進生成失敗——
    重複計數是「把兩個成因混在一起」的另一種形式。
    """
    with caplog.at_level(logging.WARNING):
        embed_dedup(
            [{"id": 1, "title": "無鍵"}, _art(2, None), _art(3, BASE)], [], threshold=0.88
        )

    warns = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("1/3" in m and "attach_embeddings" in m for m in warns), \
        f"缺鍵應報 1/3 並指名 attach_embeddings：{warns}"
    assert any("1/3" in m and "attach_embeddings" not in m for m in warns), \
        f"生成失敗應獨立報 1/3（不是 2/3）：{warns}"
