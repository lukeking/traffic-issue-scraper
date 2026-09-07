"""
Auto-KB expansion job.

Collects [[term]] markers from Supabase FFXIV articles, resolves unknown terms
via a single Gemini batch call, writes high-confidence results to the
knowledge_base table, and patches the source articles inline.

Always exits 0 — failures are logged and non-blocking.
"""
import json
import logging
import os
import re
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
    loaded = load_dotenv(override=False)
    if loaded:
        print("[dotenv] 已載入 .env（本機測試模式）", flush=True)
except ImportError:
    pass

MARKER_RE = re.compile(r"\[\[([^\]]+)\]\]")

# 標記了、但查證過**不是遊戲術語**的詞。
#
# 為什麼需要這個名單：auto_kb 每輪都會把未知詞送給 Gemini，而 Gemini 每輪都依 prompt
# 規則 2「沒把握就省略」正確地婉拒——於是這種詞永遠帶著括號留在文章裡，永遠算在
# 術語池中，**沒有任何出口**。2026-W19 的池子從 5 月躺到 8 月就是這樣來的。
# 池子的設計假設「未知 ＝ 待收錄」，但實際上有第三種狀態：根本不該被標記。
#
# ⚠️ 這裡只放**查證過確實不是術語**的詞，不要拿它當「Gemini 解不出來」的垃圾桶。
# 真的是術語但查不到譯名的（例如尚未在繁中版上線的任務名）該進 KB，不是進這裡。
IGNORED_MARKERS = {
    # 社群對某角色的俚稱，無官方譯名。上下文：「阿莉澤與 spiky boy 將作為夥伴」。
    "spiky boy",
    # LLM 用自己的話造的功能名，遊戲裡不存在這個東西。查過 Reddit 原文（1t2x42e）：
    # 討論到的是 New Game+、去宿屋看過場、Unending Codex 三者，沒有一個叫這個。
    "冒險筆記",
}

# 陸服詞 → **官方繁中**用詞的正規化表。
#
# 判準（Luke，2026-09-03）是**兩條**，不是一條：
#
# 1. **陸服詞 → 官方繁中：換。** 同一種語言、只是地區用語錯了，換過去讀者沒有損失。
# 2. **縮寫 → 全稱：不展開。** 這是不同的操作，而且會**損失**可讀性——對 FFXIV 玩家
#    來說 `FC`／`CF` 往往比官方全稱好讀。Luke 原話：「有些詞翻成繁中我反而會看不懂，
#    用英文字母縮寫才懂。」
#
# 每一條都附**逐字出處**，且出自 ffxiv.com.tw 官方頁面——社群慣用不算數，陸服維基更不算。
#
# ⚠️ **為什麼「不確定就別換」在這裡特別成立：這個改寫是單向的。** Step 5 直接覆寫
# `analysis`，`[[FC]]` 一旦變成 `公會`，標記就消失了，之後無法分辨這個「公會」原本是
# `FC`、`公會` 還是 `フリーカンパニー`。不換則隨時可以再換。所以拿不定主意時，
# 保留選擇權的那一邊嚴格佔優。
#
# ⚠️ 這張表在 Step 5 的 `replacement_map` 裡**排最後**，刻意贏過 KB 的 `known_jp`
# 與 `known_tw`——KB 有列把陸服詞當成譯名，照 KB 走等於把陸服詞寫進文章。
#
# ⚠️ **不要改成「把 known_jp / known_tw 的合併順序翻過來」**——該替代方案已否決。
# 那批 jp→tw 列多數是 `auto_generated=True`（LLM 寫的），而在正好這批樣本上正確率是
# 3 取 1（`地下城`→`迷宮` 對，`FC`→`部隊`、`CF`→`隨機任務` 都錯）。翻順序等於授權
# LLM 產生的列全面改寫文章正文；這張人工查證過的明示表才是那個決定。
#
# ── 刻意不在表裡的詞，以及各自的重開條件 ─────────────────────────────────
#
# `FC`／`CF`：**依規則 2 不展開。** 官方繁中查得到（`公會`／`任務搜尋器`，見下方
#   KB 附註），所以擋住它們的不是「查不到」而是規則 2。**重開條件：哪天讀到某個縮寫
#   覺得展開比較好，就把那一個加進來並附出處——不要整批展開。**
#   ⚠️ 目前 `[[FC]]`／`[[CF]]` 解析成 `FC`／`CF` 是靠 KB 的 `known_tw`
#   （id=95 `コンテンツファインダー→CF`、id=122 `フリーカンパニー→FC`）贏過
#   `known_jp`（id=147 `FC→公會`、id=150 `CF→任務搜尋器`，2026-09-03 已更正為官方
#   繁中）。**那兩列存官方全稱是刻意的**——KB 要回答「FC 是什麼意思」給 LLM 當語境，
#   和「文章正文顯示什麼」是兩件事。測試鎖住了這個行為。
#
# `極本`：**依規則 1 不換，因為查不到。** KB 第 144 列把它對到 `極神`，但官方繁中用的是
#   **前綴構詞**——「極 澤蓮尼亞殲滅戰」、「極 佐拉加殲滅戰」，而 `極神` 與 `極本` 在官方
#   補丁說明裡**都出現 0 次**。沒有官方單詞對應就不正規化。**重開條件：官方出現單詞用法。**
TW_NORMALISATION: dict[str, str] = {
    # 陸服詞。官方繁中補丁說明逐字：「追加全新迷宮挑戰「王城遺跡永護塔底」。」；
    # 該份補丁說明中「地下城」出現 0 次。
    # 來源 https://www.ffxiv.com.tw/web/special/patchnote_log/patch_7.2_notes.html
    "地下城": "迷宮",
}

SYSTEM_PROMPT = "你是 FFXIV 術語翻譯專家，專精繁體中文（台灣）玩家社群用語。"

# 規則 4 的 category 白名單：2026-08-31 對實際資料校準過一次。
#
# 校準前是 14 個值，而表裡有 21 個值在用——**但 LLM 從未違反過規則 4**：
# 46 列不合的全部 `auto_generated=0`，來自 spec 005 當初從 knowledge-base.md 遷移的
# 既有詞彙（45 列）＋當天手加的 `角色縮寫`（1 列）。所以問題不是「規則沒有執行者」，
# 是**分類法分家**：同一個概念兩套詞。最清楚的一對是 `地區`（白名單裡、0 列）
# vs `地點`（資料裡 2 列）；而 `職業縮寫` 有 23 列卻不在白名單，LLM 遇到職業縮寫
# 只能標成 `職業`，與那 23 列分岔。
#
# 所以修法是**補齊白名單**，不是刪掉規則 4——白名單是唯一讓 LLM 輸出保持一致的東西，
# 刪了它 LLM 會自由發明類別，分家只會更嚴重。（當天第一個念頭是刪掉它，已推翻。）
# `地區` 改成 `地點`：資料那側有列，改白名單比改資料便宜。
# `技能` 目前 0 列但是合理的類別，留著。
#
# ⚠️ 這條白名單**沒有執行者**（寫入處是 `entry.get("category") or ""`，DB 也無 constraint），
# `必須` 讀起來像契約但它只是對 LLM 的請求。要加 enum 檢查是另一件事，刻意未做——
# 目前 `category` 的唯一用途是 analyzer.py 把它渲染進餵給 LLM 的表格當語境，
# 沒有任何邏輯以它分支，所以一個沒見過的值不會壞掉任何東西，只會讓分類法更散。
USER_PROMPT_TMPL = """\
以下是尚未收錄於知識庫的 FFXIV 術語，請為每個術語提供繁體中文（台灣）翻譯。

規則：
1. 只回傳你有高把握的術語（來源：官方 TW 補丁說明 > TW 維基/社群慣用）
2. 對沒把握或資訊不足的術語，直接省略，不要猜測
3. 回傳嚴格 JSON 陣列格式，每個元素包含：jp_term, tw_term, en_term, category, notes
4. category 必須是以下之一：遊戲、資料片、資料片縮寫、副本、副本縮寫、職能、職業、職業縮寫、技能、道具、裝備、貨幣、地點、系統、功能、功能縮寫、機制、角色、角色縮寫、任務、社交、公告、更新
5. 若無任何把握的術語，回傳空陣列 []

待翻譯術語（每行一個）：
{terms_list}

回傳格式範例：
[
  {{"jp_term": "暁月のフィナーレ", "tw_term": "曉月", "en_term": "Endwalker", "category": "資料片", "notes": "6.0（2021）"}}
]
"""


def call_gemini(terms: list[str], api_key: str, model_name: str) -> list[dict]:
    terms_list = "\n".join(terms)
    user_prompt = USER_PROMPT_TMPL.format(terms_list=terms_list)

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}"
        f":generateContent"
    )
    headers = {"x-goog-api-key": api_key}
    payload = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"parts": [{"text": user_prompt}]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 4096},
    }

    for attempt in range(3):
        try:
            import requests as req
            resp = req.post(url, json=payload, headers=headers, timeout=60)
            if resp.status_code >= 500:
                logger.warning("Gemini 回應 %d，第 %d 次重試", resp.status_code, attempt + 1)
                time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            raw = resp.json()
            text = raw["candidates"][0]["content"]["parts"][0]["text"]
            # Strip markdown code fences if present
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
            return json.loads(text)
        except Exception as exc:
            logger.warning("Gemini 呼叫失敗（第 %d 次）：%s", attempt + 1, exc)
            if attempt < 2:
                time.sleep(2 ** attempt)

    logger.error("Gemini 所有重試均失敗，跳過本次 KB 擴充")
    return []


def main() -> None:
    url = (os.environ.get("SUPABASE_URL") or "").strip()
    key = (os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_KEY") or "").strip()
    gemini_api_key = (os.environ.get("GEMINI_API_KEY") or "").strip()
    gemini_model = (os.environ.get("GEMINI_MODEL_NAME") or "gemini-2.5-flash").strip()

    if not url or not key:
        logger.error("SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY 未設定，中止")
        sys.exit(0)
    if not gemini_api_key:
        # ⚠️ 同一個形狀的閘：修補（Step 5）其實不需要 Gemini，所以嚴格說這道閘也會
        # 連帶擋掉修補。刻意保留 fail-fast——CI 一定會設這個 key，而缺 key 通常
        # 代表環境配錯，此時「安靜地只做一半」比中止更難察覺。要放寬的話改成
        # warning + 繼續即可，但那是獨立的決定。
        logger.error("GEMINI_API_KEY 未設定，中止")
        sys.exit(0)

    from supabase import create_client
    supabase = create_client(url, key)

    # Step 1 — load existing KB terms (both jp_term and tw_term)
    # tw_term is needed so [[Chinese term]] markers written by the pipeline
    # aren't mistakenly sent to Gemini as unknown Japanese terms.
    try:
        kb_result = supabase.table("knowledge_base").select("jp_term, tw_term").execute()
        kb_rows: list[dict] = kb_result.data or []  # type: ignore[assignment]
        existing_jp: set[str] = {str(r["jp_term"]) for r in kb_rows}
        # tw_term→tw_term map lets Step 5 strip brackets from [[TW term]] that are
        # already translated; value equals key because no further mapping is needed.
        known_tw: dict[str, str] = {str(r["tw_term"]): str(r["tw_term"]) for r in kb_rows}
        # jp_term→tw_term for the WHOLE KB, not just this run's Gemini output.
        # Without this a term added to the KB *after* its [[marker]] was written is
        # unfixable: Step 2 filters it out as "known" so Gemini never sees it again,
        # and Step 5 never had it in the map — the marker stays bracketed forever.
        # (Latent until 2026-08-31, when `ララ` was added by hand and became the first
        # instance. Measured then: 419 ffxiv articles, 4 bracketed markers, 1 stuck.)
        known_jp: dict[str, str] = {str(r["jp_term"]): str(r["tw_term"]) for r in kb_rows}
        existing_terms: set[str] = existing_jp | set(known_tw)
        logger.info("知識庫現有術語：%d 個（JP）＋ %d 個（TW）", len(existing_jp), len(known_tw))
    except Exception as exc:
        logger.error("無法讀取 knowledge_base 表格：%s", exc)
        sys.exit(0)

    # Step 2 — collect [[term]] misses from FFXIV articles
    # Fetch all FFXIV articles and filter for [[markers]] client-side;
    # supabase-py quotes column names in .like(), breaking JSONB cast syntax.
    try:
        articles_result = (
            supabase.table("articles")
            .select("id, analysis")
            .eq("content_type", "ffxiv")
            .execute()
        )
        all_ffxiv: list[dict] = articles_result.data or []  # type: ignore[assignment]
        article_rows = [r for r in all_ffxiv if "[[" in json.dumps(r.get("analysis") or {}, ensure_ascii=False)]
    except Exception as exc:
        logger.error("無法查詢 articles 表格：%s", exc)
        sys.exit(0)

    unknown_terms: set[str] = set()
    for row in article_rows:
        text = json.dumps(row.get("analysis") or {}, ensure_ascii=False)
        for m in MARKER_RE.finditer(text):
            term = m.group(1).strip()
            if term not in existing_terms and term not in IGNORED_MARKERS:
                unknown_terms.add(term)

    # ⚠️ 這裡曾經是 `sys.exit(0)`，而那讓 Step 5 的修補**永遠到不了**。
    # 起初無害：Step 5 只處理「本輪 Gemini 剛解出來的詞」，沒有未知詞就真的沒事做。
    # 但 Step 5 後來長出兩個獨立於 Gemini 的職責——修補全 KB 的 jp→tw、拆掉
    # IGNORED_MARKERS 的括號——而那兩者恰好會把未知集清空，於是它們自己觸發了
    # 這個早退，把自己關在門外。2026-08-31 實測：三個標記全部「已知或忽略」→
    # log 印「沒有未知術語需要解析，結束」→ 一列都沒被修補。
    # 所以未知集為空只該跳過 Gemini（Step 3/4），不該跳過修補（Step 5）。
    if not unknown_terms:
        logger.info("沒有未知術語需要解析，跳過 Gemini；Step 5 的修補仍會執行")
        gemini_entries: list[dict] = []
    else:
        logger.info("發現未知術語 %d 個：%s", len(unknown_terms), ", ".join(sorted(unknown_terms)))
        # Step 3 — call Gemini
        gemini_entries = call_gemini(list(unknown_terms), gemini_api_key, gemini_model)

    # Step 4 — insert valid entries
    newly_added: dict[str, str] = {}  # jp_term -> tw_term
    for entry in gemini_entries:
        jp = (entry.get("jp_term") or "").strip()
        tw = (entry.get("tw_term") or "").strip()
        if not jp or not tw:
            logger.warning("[KB AUTO-MISS] 跳過無效 Gemini 回應項目：%s", entry)
            continue
        if jp in existing_jp:
            logger.info("術語 %s 已存在，跳過", jp)
            continue
        try:
            supabase.table("knowledge_base").insert({
                "jp_term": jp,
                "tw_term": tw,
                "en_term": entry.get("en_term") or "",
                "category": entry.get("category") or "",
                "notes": entry.get("notes") or "",
                "auto_generated": True,
            }).execute()
            newly_added[jp] = tw
            logger.info("[KB AUTO] 新增術語：%s → %s", jp, tw)
        except Exception as exc:
            logger.warning("無法寫入術語 %s：%s", jp, exc)

    # Step 5 — inline re-resolution
    # replacement_map covers: (a) newly Gemini-resolved jp→tw, (b) already-known
    # tw→tw so [[TW term]] brackets get stripped, (c) every KB jp→tw, so a
    # manually-added term repairs markers written before it existed, and
    # (d) TW_NORMALISATION, merged LAST so it outranks every KB-derived entry.
    #
    # ⚠️ Merge order between known_jp and known_tw is load-bearing and deliberately
    # keeps known_jp FIRST, i.e. known_tw still wins on collisions. 25 strings are
    # both some row's jp_term and another row's tw_term; for 21 of them the jp row's
    # tw_term IS the string, so order cannot matter. The 4 that would change are
    # CF→隨機任務, FC→部隊, 地下城→迷宮, 極本→極神. Flipping the order was proposed
    # and is REJECTED: those jp→tw rows are largely auto_generated=True (LLM-written)
    # and on this very sample only 1 of 3 was correct, so flipping would grant
    # LLM-generated rows blanket authority to rewrite article text.
    # The content question those 4 raised is now ANSWERED, by hand and explicitly —
    # see TW_NORMALISATION at the top of this file: 地下城/FC/CF normalise to the
    # official TW word (sources cited there) and 極本 is deliberately excluded
    # because no official single-word equivalent exists. No longer an open question.
    # ignored 排在 known_* 之後：若某個詞同時出現在兩邊那是矛盾，以本檔明示的決定為準。
    # 值等於鍵 ＝ 只拆括號、保留原文（那些詞都嵌在句子裡，整段移除會讀不通）。
    ignored_map: dict[str, str] = {t: t for t in IGNORED_MARKERS}
    replacement_map: dict[str, str] = {
        **known_jp, **known_tw, **ignored_map, **newly_added, **TW_NORMALISATION,
    }
    all_miss_terms: set[str] = set()

    for row in article_rows:
        text = json.dumps(row.get("analysis") or {}, ensure_ascii=False)
        replaced = False

        def replace_marker(m: re.Match) -> str:
            nonlocal replaced
            term = m.group(1).strip()
            if term in replacement_map:
                replaced = True
                return replacement_map[term]
            return m.group(0)

        patched_text = MARKER_RE.sub(replace_marker, text)

        if replaced:
            try:
                patched_analysis = json.loads(patched_text)
                article_id = row["id"]  # type: ignore[index]
                supabase.table("articles").update({"analysis": patched_analysis}).eq("id", article_id).execute()
                logger.info("文章 %s 已內聯修補術語", article_id)
            except Exception as exc:
                logger.warning("無法更新文章 %s：%s", row["id"], exc)

        # collect still-unresolved terms in this article
        for m in MARKER_RE.finditer(patched_text):
            all_miss_terms.add(m.group(1).strip())

    if all_miss_terms:
        logger.warning("========== ⚠️  KB AUTO-MISS 術語待審查 ==========")
        logger.warning("以下術語在本次 Gemini 解析中無法確認翻譯：")
        for t in sorted(all_miss_terms):
            logger.warning("  • %s", t)
        logger.warning("術語已顯示於 FFXIV 頁面術語池，等待知識庫收錄。")
        logger.warning("==============================================")

    logger.info("Auto-KB 完成：新增 %d 個術語，%d 個術語仍待審查",
                len(newly_added), len(all_miss_terms))


if __name__ == "__main__":
    main()
